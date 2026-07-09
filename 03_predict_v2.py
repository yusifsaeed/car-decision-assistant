"""
Predict the price of a single car using the v2 model + encodings.
Now returns a price RANGE (10th-90th percentile) alongside the point estimate.
Edit the `car` dict below and run: python3 03_predict_v2.py
"""
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
    car_age = max(0, 2026 - car['Year'])
    row['CarAge'] = car_age
    row['Mileage'] = car['Mileage']
    row['HasEngineCC'] = int(car.get('EngineCC') is not None)
    row['EngineCC'] = car.get('EngineCC') if car.get('EngineCC') is not None else 1600
    row['MileagePerYear'] = min(car['Mileage'] / max(car_age, 1), 100_000)

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
