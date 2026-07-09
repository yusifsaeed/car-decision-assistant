"""
Predict the price of a single car using the v2 model + encodings.
Now returns a price RANGE (10th-90th percentile) alongside the point estimate.
Edit the `car` dict below and run: python3 03_predict_v2.py
"""
from datetime import date
import pandas as pd
import numpy as np
import joblib

model = joblib.load('/home/claude/car_price_model/best_model_v2.pkl')
quantile_models = joblib.load('/home/claude/car_price_model/quantile_models_v2.pkl')
feature_columns = joblib.load('/home/claude/car_price_model/feature_columns_v2.pkl')
brand_enc = joblib.load('/home/claude/car_price_model/Brand_encoding.pkl')
model_enc = joblib.load('/home/claude/car_price_model/Model_encoding.pkl')

# median listing count fallback for models the training data barely saw
DEFAULT_LISTING_COUNT = 5

# BUGFIX: previously build_row() computed car_age = max(0, 2026 - car['Year']), hardcoded.
# That both goes stale year over year AND collapses any Year >= 2026 to the same CarAge=0,
# so e.g. 2026 and 2027 inputs produced identical predictions (CarAge is ~48% of feature
# importance). Using the real current year fixes the staleness. It does NOT fix the
# collapse - the model itself was trained on CarAge clipped the same way - so callers
# should treat Year > CURRENT_YEAR as "not yet supported" until the model is retrained
# with finer-grained CarAge (see 01_clean_merge_v2.py / 02_train_model_v2.py notes).
CURRENT_YEAR = date.today().year

car = {
    'Brand': 'BMW',
    'Model': '318i',
    'Year': 2020,
    'Mileage': 80_000,
    'Transmission': 'Automatic',
    'FuelType': 'Gas',
    'EngineCC': 1500,
    'Source': 'ContactCars',
    'Governorate': 'Cairo',        # 'Unknown' if not known
}


def build_row(car: dict) -> pd.DataFrame:
    row = {}
    if car['Year'] > CURRENT_YEAR:
        raise ValueError(
            f"Year {car['Year']} is beyond {CURRENT_YEAR}: the model cannot yet distinguish "
            f"next-year models from {CURRENT_YEAR}-model cars (see BUGFIX note above). "
            f"Use Year <= {CURRENT_YEAR} until the model is retrained."
        )
    car_age = max(0, CURRENT_YEAR - car['Year'])
    row['CarAge'] = car_age
    row['Mileage'] = car['Mileage']
    row['HasEngineCC'] = int(car.get('EngineCC') is not None)
    row['EngineCC'] = car.get('EngineCC') if car.get('EngineCC') is not None else 1600

    row['Brand_enc'] = brand_enc['mapping'].get(car['Brand'], brand_enc['global_mean'])
    row['Model_enc'] = model_enc['mapping'].get(car['Model'], model_enc['global_mean'])
    row['ModelListingCount'] = DEFAULT_LISTING_COUNT  # unknown at inference time; neutral default

    row['Transmission_Manual'] = int(car['Transmission'] == 'Manual')
    for ft in ['Electric', 'Gas', 'Hybrid', 'Natural Gas', 'Plug-in Hybrid', 'Unknown']:
        row[f'FuelType_{ft}'] = int(car.get('FuelType', 'Unknown') == ft)
    row['Source_Hatla2ee'] = int(car['Source'] == 'Hatla2ee')

    gov = car.get('Governorate', 'Unknown')
    for col in feature_columns:
        if col.startswith('Governorate_'):
            row[col] = int(col == f'Governorate_{gov}')

    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)


def predict_price(car: dict):
    X = build_row(car)
    point = float(np.expm1(model.predict(X)[0]))
    low = float(np.expm1(quantile_models[0.1].predict(X)[0]))
    high = float(np.expm1(quantile_models[0.9].predict(X)[0]))
    return point, low, high


if __name__ == '__main__':
    point, low, high = predict_price(car)
    print(f"Predicted price for {car['Year']} {car['Brand']} {car['Model']}:")
    print(f"  Best estimate: {point:,.0f} EGP")
    print(f"  Likely range (10th-90th pct): {low:,.0f} - {high:,.0f} EGP")
