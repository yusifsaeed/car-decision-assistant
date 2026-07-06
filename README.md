# Car Price Predictor 🚗

A Streamlit web app that predicts the market price of a used car in Egypt, trained on merged listings from **ContactCars** and **Hatla2ee**.

## Live demo
Once deployed on Streamlit Community Cloud, put your link here, e.g.:
`https://your-app-name.streamlit.app`

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Model
- **Algorithm:** XGBoost Regressor (log-price target)
- **Data:** ~19.6K cleaned listings after removing duplicates, unrealistic values, and corrupted rows
- **Performance:** R² ≈ 0.69–0.77, MAE ≈ 375–410K EGP
- **Features:** Brand & Model (target-encoded), car age, mileage, transmission, fuel type, engine size

## Files
| File | Purpose |
|---|---|
| `app.py` | Streamlit UI |
| `best_model.pkl` | Trained XGBoost model |
| `feature_columns.pkl` | Expected model input columns |
| `Brand_encoding.pkl` / `Model_encoding.pkl` | Target-encoding lookups |
| `brand_models.json` | Brand → Model dropdown data |
| `01_clean_merge.py` / `02_train_model.py` | Training pipeline (for reference/retraining) |
