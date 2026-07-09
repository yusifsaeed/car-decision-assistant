# Car Price Predictor (v2) 🚗

Streamlit app predicting used/new car prices in Egypt, trained on merged ContactCars + Hatla2ee listings.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's new in v2
- Trained on a larger, cleaner single merged dataset (~19.4K rows)
- Added **City** (extracted from Location) as a model feature
- Added **New vs Used** as a model feature
- Monotonic constraints: price can never increase with higher mileage or older age
- Performance improved: R² ≈ 0.77, MAE ≈ 357K EGP, MAPE ≈ 24%

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `best_model.pkl` | Trained XGBoost model (monotonic constraints) |
| `feature_columns.pkl` | Expected model input columns |
| `Brand_encoding.pkl` / `Model_encoding.pkl` / `City_encoding.pkl` | Target-encoding lookups |
| `brand_models.json` | Brand → Model and City dropdown data |
| `01_clean.py` / `02_train_model.py` | Training pipeline (for reference/retraining) |
