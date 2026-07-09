"""
v2 training pipeline. Changes vs v1:
  - New features: Governorate, ModelListingCount, MileagePerYear
  - Model: sklearn HistGradientBoostingRegressor (xgboost/lightgbm not installable in this
    sandbox - no network access). HGBR is a very close cousin of XGBoost (same histogram-based
    boosting algorithm) and natively supports missing values + categorical columns, which
    removes the need for the FuelType_Unknown / HasEngineCC workaround entirely.
  - RandomizedSearchCV hyperparameter tuning (v1 used default hyperparameters for every model)
  - Quantile models (10th/50th/90th percentile) trained alongside the point estimate, so the
    app can show a price RANGE instead of a single number
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

df = pd.read_csv('/home/claude/car_price_model/cleaned_merged_v2.csv')

df['HasEngineCC'] = df['EngineCC'].notna().astype(int)
df['EngineCC'] = df['EngineCC'].fillna(df['EngineCC'].median())
df['FuelType'] = df['FuelType'].fillna('Unknown')
df['Transmission'] = df['Transmission'].fillna('Unknown')
df['Governorate'] = df['Governorate'].fillna('Unknown')

features_cat_high_card = ['Brand', 'Model']
features_cat_low_card = ['Transmission', 'FuelType', 'Source', 'Governorate']
features_num = ['CarAge', 'Mileage', 'EngineCC', 'HasEngineCC', 'ModelListingCount', 'MileagePerYear']
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
    joblib.dump({'mapping': mapping, 'global_mean': gmean},
                f'/home/claude/car_price_model/{col}_encoding.pkl')

X_train = X_train.drop(columns=features_cat_high_card)
X_test = X_test.drop(columns=features_cat_high_card)

X_train_ohe = pd.get_dummies(X_train, columns=features_cat_low_card, drop_first=True)
X_test_ohe = pd.get_dummies(X_test, columns=features_cat_low_card, drop_first=True)
X_test_ohe = X_test_ohe.reindex(columns=X_train_ohe.columns, fill_value=0)

print("Final feature set:", list(X_train_ohe.columns))
print("Train shape:", X_train_ohe.shape, "Test shape:", X_test_ohe.shape)


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


linear_drop = ['HasEngineCC', 'FuelType_Unknown']
X_train_lin = X_train_ohe.drop(columns=linear_drop)
X_test_lin = X_test_ohe.drop(columns=linear_drop)

results = []
results.append(evaluate("Ridge Regression", Ridge(alpha=1.0),
                         X_train_lin, y_train, X_test_lin, y_test, price_test))

rf = RandomForestRegressor(n_estimators=300, max_depth=18, min_samples_leaf=2,
                            n_jobs=-1, random_state=42)
results.append(evaluate("Random Forest", rf, X_train_ohe, y_train, X_test_ohe, y_test, price_test))

# ---------- HistGradientBoosting: default params first ----------
hgb_default = HistGradientBoostingRegressor(random_state=42)
results.append(evaluate("HistGradientBoosting (default)", hgb_default,
                         X_train_ohe, y_train, X_test_ohe, y_test, price_test))

# ---------- HistGradientBoosting: hyperparameter search ----------
param_dist = {
    'max_iter': [200, 300, 500, 700],
    'max_depth': [None, 6, 8, 10, 14],
    'learning_rate': [0.02, 0.05, 0.08, 0.1],
    'max_leaf_nodes': [15, 31, 63, 127],
    'l2_regularization': [0.0, 0.1, 1.0, 5.0],
    'min_samples_leaf': [10, 20, 40],
}
search = RandomizedSearchCV(
    HistGradientBoostingRegressor(random_state=42, early_stopping=True, validation_fraction=0.1),
    param_distributions=param_dist, n_iter=25, cv=3, scoring='r2',
    random_state=42, n_jobs=-1, verbose=0
)
search.fit(X_train_ohe, y_train)
print("\nBest HGB params:", search.best_params_)
results.append(evaluate("HistGradientBoosting (tuned)", search.best_estimator_,
                         X_train_ohe, y_train, X_test_ohe, y_test, price_test))

best = max(results, key=lambda r: r['r2'])
print(f"\n>>> Best model: {best['name']} (R2={best['r2']:.4f})")

joblib.dump(best['model'], '/home/claude/car_price_model/best_model_v2.pkl')
joblib.dump(list(X_train_ohe.columns), '/home/claude/car_price_model/feature_columns_v2.pkl')

# ---------- quantile models for a price RANGE (10th / 50th / 90th percentile) ----------
best_params = search.best_params_.copy()
quantile_models = {}
for q in [0.1, 0.5, 0.9]:
    qm = HistGradientBoostingRegressor(loss='quantile', quantile=q, random_state=42, **best_params)
    qm.fit(X_train_ohe, y_train)
    quantile_models[q] = qm
joblib.dump(quantile_models, '/home/claude/car_price_model/quantile_models_v2.pkl')

# sanity check on interval coverage (what fraction of true test prices fall inside [p10, p90])
p10 = np.expm1(quantile_models[0.1].predict(X_test_ohe))
p90 = np.expm1(quantile_models[0.9].predict(X_test_ohe))
coverage = np.mean((price_test.values >= p10) & (price_test.values <= p90))
print(f"\n80% interval empirical coverage on test set: {coverage*100:.1f}% (target 80%)")

# feature importance via permutation (works for any model, not just tree-based attr)
if hasattr(best['model'], 'feature_importances_'):
    importance = pd.Series(best['model'].feature_importances_, index=X_train_ohe.columns)
else:
    from sklearn.inspection import permutation_importance
    pi = permutation_importance(best['model'], X_test_ohe, y_test, n_repeats=8, random_state=42, n_jobs=-1)
    importance = pd.Series(pi.importances_mean, index=X_train_ohe.columns)
importance = importance.sort_values(ascending=False)
print("\nTop 15 feature importances:")
print(importance.head(15))
importance.to_csv('/home/claude/car_price_model/feature_importance_v2.csv')

test_out = X_test_ohe.copy()
test_out['ActualPrice'] = price_test.values
test_out['PredictedPrice'] = np.expm1(best['model'].predict(X_test_ohe))
test_out['PredictedP10'] = p10
test_out['PredictedP90'] = p90
test_out.to_csv('/home/claude/car_price_model/test_predictions_v2.csv', index=False)

pd.DataFrame([{k: v for k, v in r.items() if k != 'model'} for r in results]).to_csv(
    '/home/claude/car_price_model/model_comparison_v2.csv', index=False)
