import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.markdown("""
<style>
.big-price { font-size: 2.6rem; font-weight: 800; color: #16a34a; text-align: center; margin: 0.5rem 0; }
.price-range { font-size: 1.1rem; font-weight: 600; color: #374151; text-align: center; margin-bottom: 0.25rem; }
.price-caption { text-align: center; color: #6b7280; margin-bottom: 1.5rem; }
.stButton>button { width: 100%; height: 3rem; font-size: 1.1rem; font-weight: 600; border-radius: 0.6rem; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model_v2.pkl')
    quantile_models = joblib.load('quantile_models_v2.pkl')
    feature_columns = joblib.load('feature_columns_v2.pkl')
    brand_enc = joblib.load('Brand_encoding.pkl')
    model_enc = joblib.load('Model_encoding.pkl')
    with open('brand_models.json', encoding='utf-8') as f:
        lookup = json.load(f)
    return model, quantile_models, feature_columns, brand_enc, model_enc, lookup


model, quantile_models, feature_columns, brand_enc, model_enc, lookup = load_artifacts()
brands = lookup['brands']
brand_models = lookup['brand_models']

FUEL_TYPES = ['Gas', 'Diesel', 'Electric', 'Hybrid', 'Plug-in Hybrid', 'Natural Gas', 'Unknown']
GOVERNORATES = sorted(c.replace('Governorate_', '') for c in feature_columns if c.startswith('Governorate_'))
CURRENT_YEAR = 2026
DEFAULT_LISTING_COUNT = 5


def build_row(brand, model_name, year, mileage, transmission, fuel_type, engine_cc, source, governorate):
    row = {}
    car_age = max(0, CURRENT_YEAR - year)
    row['CarAge'] = car_age
    row['Mileage'] = mileage
    row['HasEngineCC'] = int(engine_cc is not None)
    row['EngineCC'] = engine_cc if engine_cc else 1600

    row['Brand_enc'] = brand_enc['mapping'].get(brand, brand_enc['global_mean'])
    row['Model_enc'] = model_enc['mapping'].get(model_name, model_enc['global_mean'])
    row['ModelListingCount'] = DEFAULT_LISTING_COUNT

    row['Transmission_Manual'] = int(transmission == 'Manual')
    for ft in ['Electric', 'Gas', 'Hybrid', 'Natural Gas', 'Plug-in Hybrid', 'Unknown']:
        row[f'FuelType_{ft}'] = int(fuel_type == ft)
    row['Source_Hatla2ee'] = int(source == 'Hatla2ee')

    for col in feature_columns:
        if col.startswith('Governorate_'):
            row[col] = int(col == f'Governorate_{governorate}')

    return pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)


def predict_price(*args):
    X = build_row(*args)
    point = float(np.expm1(model.predict(X)[0]))
    low = float(np.expm1(quantile_models[0.1].predict(X)[0]))
    high = float(np.expm1(quantile_models[0.9].predict(X)[0]))
    return point, low, high


st.title("🚗 Car Price Predictor")
st.caption("Estimate the market price of a used car in Egypt, trained on ContactCars + Hatla2ee listings.")

col1, col2 = st.columns(2)

with col1:
    brand = st.selectbox("Brand", brands, index=brands.index('Toyota') if 'Toyota' in brands else 0)
    available_models = brand_models.get(brand, [])
    model_name = st.selectbox("Model", available_models if available_models else ["Other"])
    year = st.number_input("Year", min_value=1980, max_value=2027, value=2020, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=80_000, step=1_000)

with col2:
    transmission = st.radio("Transmission", ["Automatic", "Manual"], horizontal=True)
    fuel_type = st.selectbox("Fuel type", FUEL_TYPES)
    governorate = st.selectbox("Governorate (optional)", ["Unknown"] + [g for g in GOVERNORATES if g != "Unknown"])
    has_engine = st.checkbox("I know the engine size (CC)", value=False)
    engine_cc = st.number_input("Engine CC", min_value=600, max_value=8000, value=1600, step=100) if has_engine else None

st.divider()

car_age_check = max(1, CURRENT_YEAR - year)
km_per_year = mileage / car_age_check
if km_per_year > 35_000:
    st.warning(
        f"⚠️ {mileage:,.0f} km on a {year} car works out to ~{km_per_year:,.0f} km/year, "
        "well above typical usage (~15-20K km/year). The model will price this as an "
        "unusually heavily-used car, which can make it look cheaper than a slightly older "
        "car with more typical mileage. Double check the Year and Mileage are correct.",
        icon="⚠️"
    )

if st.button("Predict price 💰"):
    point, low, high = predict_price(brand, model_name, year, mileage, transmission,
                                      fuel_type, engine_cc, 'ContactCars', governorate)
    st.markdown(f'<div class="big-price">{point:,.0f} EGP</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="price-range">Likely range: {low:,.0f} – {high:,.0f} EGP</div>',
                unsafe_allow_html=True)
    st.markdown(
        f'<div class="price-caption">Estimated price for a {year} {brand} {model_name}</div>',
        unsafe_allow_html=True
    )
    st.info("The range covers the 10th-90th percentile of similar listings in the training "
            "data (~80% of comparable cars fall in it) - it's a market reference, not an appraisal.",
            icon="ℹ️")

with st.expander("ℹ️ About this model"):
    st.write("""
    - Trained on merged listings from **ContactCars** and **Hatla2ee** (~19.1K cleaned rows,
      with per-brand outlier filtering rather than one global price band).
    - Model: **HistGradientBoostingRegressor** (sklearn's close cousin of XGBoost), tuned with
      randomized search, predicting log-price then converting back to EGP.
    - Test performance: **R² ≈ 0.84**, MAE ≈ 270K EGP, MAPE ≈ 20%.
    - New signals versus the first version: **governorate**, **how common the exact
      brand/model listing is** in the data, and **km driven per year** (usage intensity),
      plus a **price range** (10th-90th percentile) instead of a single number.
    - Brand and Model are encoded using smoothed target (mean-price) encoding, so unseen
      brands/models fall back to the overall average price.
    """)
