import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config & styling
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Car Price Predictor", page_icon="🚗", layout="centered")

st.markdown("""
<style>
.big-price {
    font-size: 2.6rem;
    font-weight: 800;
    color: #16a34a;
    text-align: center;
    margin: 0.5rem 0;
}
.price-caption {
    text-align: center;
    color: #6b7280;
    margin-bottom: 1.5rem;
}
.stButton>button {
    width: 100%;
    height: 3rem;
    font-size: 1.1rem;
    font-weight: 600;
    border-radius: 0.6rem;
}
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model artifacts (cached so they load once per session)
# ---------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load('best_model.pkl')
    feature_columns = joblib.load('feature_columns.pkl')
    brand_enc = joblib.load('Brand_encoding.pkl')
    model_enc = joblib.load('Model_encoding.pkl')
<<<<<<< HEAD
    city_enc = joblib.load('City_encoding.pkl')
    with open('brand_models.json', encoding='utf-8') as f:
        lookup = json.load(f)
    return model, feature_columns, brand_enc, model_enc, city_enc, lookup


model, feature_columns, brand_enc, model_enc, city_enc, lookup = load_artifacts()
brands = lookup['brands']
brand_models = lookup['brand_models']
cities = lookup['cities']

FUEL_TYPES = ['Gas', 'Diesel', 'Electric', 'Hybrid', 'Plug-in Hybrid', 'Natural Gas']
CURRENT_YEAR = 2026


def predict_price(brand, model_name, city, year, mileage, transmission, fuel_type, engine_cc, is_new):
=======
    with open('brand_models.json', encoding='utf-8') as f:
        lookup = json.load(f)
    return model, feature_columns, brand_enc, model_enc, lookup


model, feature_columns, brand_enc, model_enc, lookup = load_artifacts()
brands = lookup['brands']
brand_models = lookup['brand_models']

FUEL_TYPES = ['Gas', 'Diesel', 'Electric', 'Hybrid', 'Plug-in Hybrid', 'Natural Gas', 'Unknown']
CURRENT_YEAR = 2026


def predict_price(brand, model_name, year, mileage, transmission, fuel_type, engine_cc, source):
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
    row = {}
    row['CarAge'] = max(0, CURRENT_YEAR - year)
    row['Mileage'] = mileage
    row['HasEngineCC'] = int(engine_cc is not None)
<<<<<<< HEAD
    row['Engine_CC'] = engine_cc if engine_cc else 1600  # training median fallback
    row['IsNew'] = int(is_new)

    row['Brand_enc'] = brand_enc['mapping'].get(brand, brand_enc['global_mean'])
    row['Model_enc'] = model_enc['mapping'].get(model_name, model_enc['global_mean'])
    row['City_enc'] = city_enc['mapping'].get(city, city_enc['global_mean'])

    row['Transmission_Manual'] = int(transmission == 'Manual')
    for ft in ['Electric', 'Gas', 'Hybrid', 'Natural Gas', 'Plug-in Hybrid']:
        row[f'Fuel_type_{ft}'] = int(fuel_type == ft)
=======
    row['EngineCC'] = engine_cc if engine_cc else 1600  # training median fallback

    row['Brand_enc'] = brand_enc['mapping'].get(brand, brand_enc['global_mean'])
    row['Model_enc'] = model_enc['mapping'].get(model_name, model_enc['global_mean'])

    row['Transmission_Manual'] = int(transmission == 'Manual')
    for ft in ['Electric', 'Gas', 'Hybrid', 'Natural Gas', 'Plug-in Hybrid', 'Unknown']:
        row[f'FuelType_{ft}'] = int(fuel_type == ft)
    row['Source_Hatla2ee'] = int(source == 'Hatla2ee')
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2

    X = pd.DataFrame([row]).reindex(columns=feature_columns, fill_value=0)
    log_price = model.predict(X)[0]
    return float(np.expm1(log_price))


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------
<<<<<<< HEAD
st.title("🚗 Car Price Predictor")
st.caption("Estimate the market price of a car in Egypt, trained on real ContactCars + Hatla2ee listings.")
=======
st.title("🚗 Car Decision Assisstant")
st.caption("Estimate the market price of a used car in Egypt, trained on ContactCars + Hatla2ee listings.")
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2

col1, col2 = st.columns(2)

with col1:
<<<<<<< HEAD
    brand = st.selectbox("Brand", brands, index=brands.index('Toyota') if 'Toyota' in brands else 0)
    available_models = brand_models.get(brand, [])
    model_name = st.selectbox("Model", available_models if available_models else ["Other"])
    city = st.selectbox("City", cities)
    is_new = st.checkbox("This is a brand-new car", value=False)
    year = st.number_input("Year", min_value=1980, max_value=2027, value=2020, step=1)

with col2:
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000,
                               value=0 if is_new else 80_000, step=1_000, disabled=is_new)
=======
    brand = st.selectbox("Brand", brands, index=brands.index('Nissan') if 'Nissan' in brands else 0)
    available_models = brand_models.get(brand, [])
    model_name = st.selectbox("Model", available_models if available_models else ["Other"])
    year = st.number_input("Year", min_value=1980, max_value=2027, value=2020, step=1)
    mileage = st.number_input("Mileage (km)", min_value=0, max_value=500_000, value=80_000, step=1_000)

with col2:
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
    transmission = st.radio("Transmission", ["Automatic", "Manual"], horizontal=True)
    fuel_type = st.selectbox("Fuel type", FUEL_TYPES)
    has_engine = st.checkbox("I know the engine size (CC)", value=False)
    engine_cc = st.number_input("Engine CC", min_value=600, max_value=8000, value=1600, step=100) if has_engine else None

st.divider()

if st.button("Predict price 💰"):
<<<<<<< HEAD
    effective_mileage = 0 if is_new else mileage
    price = predict_price(brand, model_name, city, year, effective_mileage,
                           transmission, fuel_type, engine_cc, is_new)
=======
    price = predict_price(brand, model_name, year, mileage, transmission, fuel_type, engine_cc, source='ContactCars')
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
    st.markdown(f'<div class="big-price">{price:,.0f} EGP</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="price-caption">Estimated price for a {year} {brand} {model_name}</div>',
        unsafe_allow_html=True
    )
<<<<<<< HEAD
    st.info("This is a model estimate (typical error ≈ 24%), not an appraisal. "
=======
    st.info("This is a model estimate (typical error ≈ 22-25%), not an appraisal. "
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
            "Use it as a reference point alongside real listings.", icon="ℹ️")

with st.expander("ℹ️ About this model"):
    st.write("""
<<<<<<< HEAD
    - Trained on ~19.4K cleaned car listings (ContactCars + Hatla2ee) after removing
      duplicates, unrealistic mileage/price/engine values, and corrupted rows.
    - Model: **XGBoost Regressor**, predicting log-price then converting back to EGP.
    - Test performance: **R² ≈ 0.77**, MAE ≈ 357K EGP, MAPE ≈ 24%.
    - Price is guaranteed to never increase with higher mileage or an older car age
      (monotonic constraint), so results stay logically consistent.
    - Brand, Model, and City use smoothed target (mean-price) encoding, so unseen
      values fall back to the overall average price.
=======
    - Trained on merged listings from **ContactCars** and **Hatla2ee** (~19.6K cleaned rows after
      removing corrupted/duplicate/outlier records).
    - Model: **XGBoost Regressor**, predicting log-price then converting back to EGP.
    - Test performance: **R² ≈ 0.69–0.77**, MAE ≈ 375–410K EGP depending on the data split.
    - Brand and Model are encoded using smoothed target (mean-price) encoding, so unseen
      brands/models fall back to the overall average price.
>>>>>>> 1cdd9df46d386ff83e82a89d3d3e2a2f9ff2a1e2
    """)
