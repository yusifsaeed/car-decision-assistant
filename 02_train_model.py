"""
Train price-prediction models on the new cleaned dataset.
Adds City (target-encoded) and IsNew (Used/New) vs the previous version.
Keeps monotonic constraints on CarAge/Mileage so price never increases with age/mileage.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import joblib

df = pd.read_csv('/home/claude/car_price_v2/cleaned.csv')

df['HasEngineCC'] = df['Engine_CC'].notna().astype(int)
df['Engine_CC'] = df['Engine_CC'].fillna(df['Engine_CC'].median())
df['Fuel_type'] = df['Fuel_type'].fillna('Unknown')
df['Transmission'] = df['Transmission'].fillna('Unknown')

features_cat_high_card = ['Brand', 'Model', 'City']   # target-encode (too many categories for OHE)
features_cat_low_card = ['Transmission', 'Fuel_type']  # one-hot
features_num = ['CarAge', 'Mileage', 'Engine_CC', 'HasEngineCC', 'IsNew']

target = 'LogPrice'

X = df[features_cat_high_card + features_cat_low_card + features_num].copy()
y = df[target].copy()
price_actual = df['Price'].copy()

X_train, X_test, y_train, y_test, price_train, price_test = train_test_split(
    X, y, price_actual, test_size=0.2, random_state=42
)


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
    full_stats = train_target.groupby(train_col).agg(['mean', 'count'])
    full_map = (full_stats['mean'] * full_stats['count'] + global_mean * smoothing) / (full_stats['count'] + smoothing)
    test_encoded = test_col.map(full_map).fillna(global_mean)
    return oof.values, test_encoded.values, full_map, global_mean


for col in features_cat_high_card:
    train_enc, test_enc, mapping, gmean = kfold_target_encode(X_train[col], y_train, X_test[col])
    X_train[col + '_enc'] = train_enc
    X_test[col + '_enc'] = test_enc
    plain_mapping = {str(k): float(v) for k, v in mapping.items()}
    joblib.dump({'mapping': plain_mapping, 'global_mean': float(gmean)},
                f'/home/claude/car_price_v2/{col}_encoding.pkl')

X_train = X_train.drop(columns=features_cat_high_card)
X_test = X_test.drop(columns=features_cat_high_card)

X_train = pd.get_dummies(X_train, columns=features_cat_low_card, drop_first=True)
X_test = pd.get_dummies(X_test, columns=features_cat_low_card, drop_first=True)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

print("Final feature set:", list(X_train.columns))
print("Train shape:", X_train.shape, "Test shape:", X_test.shape)


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


# monotonic constraints: price can only stay flat or fall as CarAge/Mileage rise
mono = [-1 if col in ('CarAge', 'Mileage') else 0 for col in X_train.columns]

results = []

lin_drop = ['HasEngineCC']  # kept for tree models; not collinear here since Fuel_type is fully populated
results.append(evaluate("Ridge Regression", Ridge(alpha=1.0),
                         X_train.drop(columns=lin_drop), y_train,
                         X_test.drop(columns=lin_drop), y_test, price_test))

rf = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=2,
                            n_jobs=-1, random_state=42, monotonic_cst=mono)
results.append(evaluate("Random Forest", rf, X_train, y_train, X_test, y_test, price_test))

xgb = XGBRegressor(n_estimators=600, max_depth=8, learning_rate=0.04,
                    subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                    random_state=42, n_jobs=-1,
                    monotone_constraints=tuple(mono))
results.append(evaluate("XGBoost", xgb, X_train, y_train, X_test, y_test, price_test))

best = max(results, key=lambda r: r['r2'])
print(f"\n>>> Best model: {best['name']} (R2={best['r2']:.4f})")

joblib.dump(best['model'], '/home/claude/car_price_v2/best_model.pkl')
joblib.dump(list(X_train.columns), '/home/claude/car_price_v2/feature_columns.pkl')

if hasattr(best['model'], 'feature_importances_'):
    importance = pd.Series(best['model'].feature_importances_, index=X_train.columns)
    importance = importance.sort_values(ascending=False)
    print("\nTop 15 feature importances:")
    print(importance.head(15))
    importance.to_csv('/home/claude/car_price_v2/feature_importance.csv')

pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results]).to_csv(
    '/home/claude/car_price_v2/model_comparison.csv', index=False)
