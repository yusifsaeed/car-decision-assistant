<<<<<<< HEAD
# Car Price Predictor (v2) 🚗

Streamlit app predicting used/new car prices in Egypt, trained on merged ContactCars + Hatla2ee listings.

## Run locally
=======
# Car Price Predictor 🚗

A Streamlit web app that predicts the market price of a used car in Egypt, trained on merged listings from **ContactCars** and **Hatla2ee**.

## Live demo
Once deployed on Streamlit Community Cloud, put your link here, e.g.:
`https://your-app-name.streamlit.app`

## Run locally

>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
```bash
pip install -r requirements.txt
streamlit run app.py
```

<<<<<<< HEAD
## What's new in v2
- Trained on a larger, cleaner single merged dataset (~19.4K rows)
- Added **City** (extracted from Location) as a model feature
- Added **New vs Used** as a model feature
- Monotonic constraints: price can never increase with higher mileage or older age
- Performance improved: R² ≈ 0.77, MAE ≈ 357K EGP, MAPE ≈ 24%
=======
## Model
- **Algorithm:** XGBoost Regressor (log-price target)
- **Data:** ~19.6K cleaned listings after removing duplicates, unrealistic values, and corrupted rows
- **Performance:** R² ≈ 0.69–0.77, MAE ≈ 375–410K EGP
- **Features:** Brand & Model (target-encoded), car age, mileage, transmission, fuel type, engine size
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
<<<<<<< HEAD
| `best_model.pkl` | Trained XGBoost model (monotonic constraints) |
| `feature_columns.pkl` | Expected model input columns |
| `Brand_encoding.pkl` / `Model_encoding.pkl` / `City_encoding.pkl` | Target-encoding lookups |
| `brand_models.json` | Brand → Model and City dropdown data |
| `01_clean.py` / `02_train_model.py` | Training pipeline (for reference/retraining) |
=======
| `best_model.pkl` | Trained XGBoost model |
| `feature_columns.pkl` | Expected model input columns |
| `Brand_encoding.pkl` / `Model_encoding.pkl` | Target-encoding lookups |
| `brand_models.json` | Brand → Model dropdown data |
| `01_clean_merge.py` / `02_train_model.py` | Training pipeline (for reference/retraining) |
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
