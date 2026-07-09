"""
Predict the price of a single car using the saved XGBoost model + encodings.
Edit the `car` dict below and run: python3 03_predict.py
"""
import pandas as pd
import numpy as np
import joblib

model = joblib.load('/home/claude/car_price_model/best_model.pkl')
feature_columns = joblib.load('/home/claude/car_price_model/feature_columns.pkl')
brand_enc = joblib.load('/home/claude/car_price_model/Brand_encoding.pkl')
model_enc = joblib.load('/home/claude/car_price_model/Model_encoding.pkl')

# ---- describe the car you want a price for ----
car = {
    'Brand': 'BMW',
    'Model': '318i',
    'Year': 2020,
    'Mileage': 80_000,
    'Transmission': 'Automatic',      # 'Automatic' or 'Manual'
    'FuelType': 'Gas',                # 'Gas' / 'Diesel' / 'Electric' / 'Hybrid' / 'Plug-in Hybrid' / 'Natural Gas' / 'Unknown'
    'EngineCC': 1500,                 # leave as None if unknown
    'Source': 'ContactCars',          # 'ContactCars' or 'Hatla2ee'
}


def predict_price(car: dict) -> float:
    row = {}
    row['CarAge'] = max(0, 2026 - car['Year'])
    row['Mileage'] = car['Mileage']
    row['HasEngineCC'] = int(car.get('EngineCC') is not None)
    row['EngineCC'] = car.get('EngineCC') if car.get('EngineCC') is not None else 1600  # training median

    row['Brand_enc'] = brand_enc['mapping'].get(car['Brand'], brand_enc['global_mean'])
    row['Model_enc'] = model_enc['mapping'].get(car['Model'], model_enc['global_mean'])

    row['Transmission_Manual'] = int(car['Transmission'] == 'Manual')
    for ft in ['Electric', 'Gas', 'Hybrid', 'Natural Gas', 'Plug-in Hybrid', 'Unknown']:
        row[f'FuelType_{ft}'] = int(car.get('FuelType', 'Unknown') == ft)
    row['Source_Hatla2ee'] = int(car['Source'] == 'Hatla2ee')

    X = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    log_price = model.predict(X)[0]
    return float(np.expm1(log_price))


if __name__ == '__main__':
    price = predict_price(car)
    print(f"Predicted price for {car['Year']} {car['Brand']} {car['Model']}: {price:,.0f} EGP")
