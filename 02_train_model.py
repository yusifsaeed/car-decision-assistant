"""
Train price-prediction models on the cleaned/merged car dataset.
Target: LogPrice (log1p of Price) -> converted back to EGP for reporting.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('/home/claude/car_price_model/cleaned_merged.csv')

# ---------- features ----------
# EngineCC and FuelType/Location are mostly missing (single-source only) -> keep as
# "known/unknown" signal rather than drop them entirely.
df['HasEngineCC'] = df['EngineCC'].notna().astype(int)
df['EngineCC'] = df['EngineCC'].fillna(df['EngineCC'].median())
df['FuelType'] = df['FuelType'].fillna('Unknown')
df['Transmission'] = df['Transmission'].fillna('Unknown')

features_cat_high_card = ['Brand', 'Model']       # target-encode (too many categories for OHE)
features_cat_low_card = ['Transmission', 'FuelType', 'Source']  # one-hot
# Year dropped: perfectly collinear with CarAge.
# HasEngineCC / FuelType_Unknown are dropped from the *linear* model input further down:
# both are exact duplicates of Source (each source only fills in the fields it collects),
# but tree models handle that redundancy fine and it's still a legitimate signal, so we
# keep them for RF/XGBoost and only strip the duplicate for Ridge.
features_num = ['CarAge', 'Mileage', 'EngineCC', 'HasEngineCC']

target = 'LogPrice'

X = df[features_cat_high_card + features_cat_low_card + features_num].copy()
y = df[target].copy()
price_actual = df['Price'].copy()

X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
    X, y, price_actual, test_size=0.2, random_state=42
)

# ---------- target encoding for Brand / Model (fit on train only, KFold to avoid leakage) ----------
def kfold_target_encode(train_col, train_target, test_col, n_splits=5, smoothing=10):
    global_mean = train_target.mean()
    oof = pd.Series(index=train_col.index, dtype=float)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for tr_idx, val_idx in kf.split(train_col):
        tr_c, val_c = train_col.iloc[tr_idx], train_col.iloc[val_idx]
        tr_t = train_target.iloc[tr_idx]
        stats = tr_t.groupby(tr_c).agg(['mean', 'count'])
        smooth_map = (stats['mean'] * stats['count'] + global_mean * smoothing) / (stats['count'] + smoothing)
        oof.iloc[val_idx] = val_c.map(smooth_map).fillna(global_mean).values
    # full mapping for test set uses all training data
    full_stats = train_target.groupby(train_col).agg(['mean', 'count'])
    full_map = (full_stats['mean'] * full_stats['count'] + global_mean * smoothing) / (full_stats['count'] + smoothing)
    test_encoded = test_col.map(full_map).fillna(global_mean)
    return oof.values, test_encoded.values, full_map, global_mean

for col in features_cat_high_card:
    train_enc, test_enc, mapping, gmean = kfold_target_encode(X_train[col], y_train, X_test[col])
    X_train[col + '_enc'] = train_enc
    X_test[col + '_enc'] = test_enc
    joblib.dump({'mapping': mapping, 'global_mean': gmean}, f'/home/claude/car_price_model/{col}_encoding.pkl')

X_train = X_train.drop(columns=features_cat_high_card)
X_test = X_test.drop(columns=features_cat_high_card)

# ---------- one-hot for low-cardinality categoricals ----------
X_train = pd.get_dummies(X_train, columns=features_cat_low_card, drop_first=True)
X_test = pd.get_dummies(X_test, columns=features_cat_low_card, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Final feature set:", list(X_train.columns))
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)

# ---------- models ----------
def evaluate(name, model, X_tr, y_tr, X_te, y_te, price_te):
    model.fit(X_tr, y_tr)
    pred_log = model.predict(X_te)
    pred_price = np.expm1(pred_log)
    mae = mean_absolute_error(price_te, pred_price)
    rmse = np.sqrt(mean_squared_error(price_te, pred_price))
    r2 = r2_score(price_te, pred_price)
    mape = np.mean(np.abs((price_te - pred_price) / price_te)) * 100
    print(f"\n{name}")
    print(f"  MAE:  {mae:,.0f} EGP")
    print(f"  RMSE: {rmse:,.0f} EGP")
    print(f"  R2:   {r2:.4f}")
    print(f"  MAPE: {mape:.2f}%")
    return {'name': name, 'model': model, 'mae': mae, 'rmse': rmse, 'r2': r2, 'mape': mape}

# For the linear model only: drop columns that exactly duplicate/negate Source_Hatla2ee
# (HasEngineCC and FuelType_Unknown are each source-determined) to avoid a singular design matrix.
linear_drop = ['HasEngineCC', 'FuelType_Unknown']
X_train_lin = X_train.drop(columns=linear_drop)
X_test_lin = X_test.drop(columns=linear_drop)

results = []
results.append(evaluate("Ridge Regression", Ridge(alpha=1.0),
                         X_train_lin, y_train, X_test_lin, y_test, price_test))

rf = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=2,
                            n_jobs=-1, random_state=42,
                            monotonic_cst=[-1 if col in ('CarAge', 'Mileage') else 0
                                           for col in X_train.columns])
results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test, price_test))

xgb = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1,
                    monotone_constraints=tuple(
                        -1 if col in ('CarAge', 'Mileage') else 0 for col in X_train.columns
                    ))
results.append(evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test, price_test))

# ---------- pick best model by R2 ----------
best = max(results, key=lambda r: r['r2'])
print(f"\n>>> Best model: {best['name']} (R2={best['r2']:.4f})")

joblib.dump(best['model'], '/home/claude/car_price_model/best_model.pkl')
joblib.dump(list(X_train.columns), '/home/claude/car_price_model/feature_columns.pkl')

# feature importance (if available)
if hasattr(best['model'], 'feature_importances_'):
    importance = pd.Series(best['model'].feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importance.head(15))
    importance.to_csv('/home/claude/car_price_model/feature_importance.csv')

# save test predictions for plotting
test_out = X_test.copy()
test_out['ActualPrice'] = price_test.values
test_out['PredictedPrice'] = np.expm1(best['model'].predict(X_test))
test_out.to_csv('/home/claude/car_price_model/test_predictions.csv', index=False)

# save results summary
pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results]).to_csv(
    '/home/claude/car_price_model/model_comparison.csv', index=False)
