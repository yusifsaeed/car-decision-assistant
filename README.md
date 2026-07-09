# Car Price Predictor (v3) 🚗

Streamlit app predicting used/new car prices in Egypt, trained on merged ContactCars + Hatla2ee listings.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## What's new in v3
- Added per-Brand+Model outlier removal (IQR-based) — caught mispriced listings a
  global price bound couldn't (e.g. a few "Nissan Sunny" rows priced at 8.8M EGP)
- Deeper XGBoost trees (depth 8) for better brand/model x age interactions
- App now warns when a prediction is based on very few listings for that exact
  Brand+Model, especially for near-new cars
- Performance improved: R² ≈ 0.84, MAE ≈ 290K EGP, MAPE ≈ 21%

## Known limitation
For a budget model with very few near-new listings in the data, the prediction can
skew toward the (higher) average price of near-new cars in general, since that
segment is dominated by more expensive cars in the training data. The app flags
this with a warning when it applies.

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `best_model.pkl` | Trained XGBoost model (monotonic constraints, depth 8) |
| `feature_columns.pkl` | Expected model input columns |
| `Brand_encoding.pkl` / `Model_encoding.pkl` / `City_encoding.pkl` | Target-encoding lookups |
| `brand_models.json` | Brand → Model, City, and sample-count data |
| `01_clean.py` / `02_train_model.py` | Training pipeline (for reference/retraining) |
