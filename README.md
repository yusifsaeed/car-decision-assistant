<p align="center">
  <img src="https://img.shields.io/badge/status-active-success" alt="status">
  <img src="https://img.shields.io/badge/python-3.8%2B-blue" alt="python">
  <img src="https://img.shields.io/badge/streamlit-app-red" alt="streamlit">
  <img src="https://img.shields.io/badge/databricks-medallion-orange" alt="databricks">
</p>

# 🚗 Car Decision Assistant

**An AI-powered platform that analyzes the Egyptian used car market to help buyers find the best vehicle at fair prices.**

Combines real-time web scraping, a cloud data engineering pipeline, and machine learning to turn scattered used-car listings into a single, trustworthy price estimate — plus market dashboards to explore trends across brands, models, locations, and years.

🔗 **[Live App](https://car-decision-assistant-1.streamlit.app/)**

---

## 📋 Table of Contents

- [Overview](#-overview)
- [The Problem & Solution](#-the-problem--solution)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Power BI Dashboards](#-power-bi-dashboards)
- [Known Issues & Limitations](#-known-issues--limitations)
- [Future Work](#-future-work)
- [Team](#-team)

---

## 📊 Overview

The used car market in Egypt is fragmented across multiple platforms (ContactCars, Hatla2ee, YallaMotor), prices fluctuate without clear patterns, and buyers have no easy way to know whether a listing is a fair deal. **Car Decision Assistant** solves this by autonomously scraping listings, centralizing and cleaning them in a governed data warehouse, and serving both an ML-based price predictor and interactive analytics dashboards.

## 🎯 The Problem & Solution

| Problem | Solution |
|---|---|
| Prices change frequently with no clear pattern | Continuously refreshed, cleaned market data |
| Listings scattered across multiple platforms | Centralized, multi-source data pipeline |
| Manual comparison is slow and unreliable | AI-powered price prediction in seconds |
| No data-driven guidance on buying decisions | Market trend dashboards + price estimator |
| Purchases based on assumptions, not data | Statistically-grounded price ranges, not guesses |

**Target users:** individual car buyers, car dealers, automotive analysts, market researchers.

## 🏗️ Architecture

Built on **Databricks** using a **Medallion Architecture** (Bronze → Silver → Gold):

```
┌─────────────────┐     ┌───────────────────────────────────────┐     ┌──────────────────────┐
│  1. DATA SOURCES │     │        2. ETL PIPELINE (Databricks)     │     │  3. OUTPUT/CONSUMPTION │
│                  │     │                                         │     │                       │
│  ContactCars     │────▶│  🥉 BRONZE   →  🥈 SILVER  →  🥇 GOLD    │────▶│  Streamlit Dashboard  │
│  Hatla2ee        │     │  Raw, as-is     Cleaned &     Business-  │     │  Power BI Reports     │
│  YallaMotor      │     │  from Selenium  validated     ready,     │     │  AI Model (price      │
│                  │     │                 & standard.   curated    │     │  prediction)          │
│  (Selenium       │     │                                         │     │                       │
│   web scraping)  │     │  Delta Lake · PySpark · Unity Catalog ·  │     │                       │
│                  │     │  Databricks Jobs · MLflow                │     │                       │
└─────────────────┘     └───────────────────────────────────────┘     └──────────────────────┘
```

| Layer | Purpose |
|---|---|
| 🥉 **Bronze** | Raw ingestion of scraped CSVs (ContactCars + Hatla2ee) as-is, schema-on-read, immutable historical archive |
| 🥈 **Silver** | Cleaning, deduplication, type casting, date standardization (`YYYY-MM-DD`), data quality checks |
| 🥇 **Gold** | Statistical outlier removal (per Brand/Model, price ≤ mean + 3×stddev), aggregations by Brand/Model/Year/Location/FuelType — business- and AI-ready |

### Web scraping challenges overcome
Dynamic website structures, infinite scrolling, frequent HTML changes, and Cloudflare anti-bot protection — handled via Selenium + `undetected-chromedriver` with scheduled, automated runs.

### Data challenges overcome
Missing values, duplicate listings, inconsistent formats, and multi-source schema differences between ContactCars and Hatla2ee.

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| **Data Ingestion** | Selenium, undetected-chromedriver, Pandas |
| **Data Warehouse** | Databricks, Delta Lake |
| **Data Processing** | PySpark, SQL |
| **ML / Modeling** | scikit-learn (HistGradientBoostingRegressor), MLflow |
| **Frontend** | Streamlit |
| **Visualization** | Power BI |
| **Governance** | Unity Catalog |
| **Scheduling** | Databricks Jobs |

## 📁 Project Structure

```
NHA-4-084/
├── Data/                       # Raw scraped datasets (Data_ContactCars.csv, hatla2ee.csv)
├── ETL/                        # Databricks notebooks: Bronze_layer, Silver_layer, Gold_layer
│   ├── Bronze_layer.ipynb      # Raw ingestion + union of both sources
│   ├── Silver_layer.ipynb      # Cleaning, validation, standardization
│   └── Gold_layer.ipynb        # Outlier filtering + business aggregations
├── Model/                      # Price prediction model pipeline
│   ├── 01_clean_merge_v2.py    # Local clean/merge + feature engineering
│   ├── 02_train_model_v2.py    # Model training, tuning, quantile models
│   ├── 03_predict_v2.py        # CLI single-prediction script
│   ├── app_v2.py               # Streamlit web app
│   ├── best_model_v2.pkl
│   ├── quantile_models_v2.pkl
│   ├── feature_columns_v2.pkl
│   ├── Brand_encoding.pkl / Model_encoding.pkl
│   ├── brand_models.json
│   ├── feature_importance_v2.csv
│   └── model_comparison_v2.csv
├── PowerBI/                     # Power BI report files (.pbix)
├── Presentation/                 # Project presentation deck
├── ScrapingCode/                 # Selenium scraping scripts for all 3 sources
├── ScreenShots/                  # App/dashboard screenshots
├── venv/                         # Local virtual environment (not committed ideally)
├── requirements.txt
└── README.md
```

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Databricks workspace access (for running the ETL notebooks)
- Git

### Installation

```bash
git clone https://github.com/nhahub/NHA-4-084.git
cd NHA-4-084

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

## 💡 Usage

### Run the Streamlit app locally
```bash
cd Model
streamlit run app_v2.py
```
Pick a brand, model, year, mileage, transmission, fuel type, and (optionally) governorate — get an estimated price plus a likely 10th–90th percentile range.

### Retrain the model
Requires the raw `Data_ContactCars.csv` and `hatla2ee.csv` (in `Data/`):
```bash
cd Model
python3 01_clean_merge_v2.py   # clean, merge, feature-engineer
python3 02_train_model_v2.py   # train, tune, evaluate, save artifacts
```

### Single CLI prediction
```bash
python3 03_predict_v2.py
```

### Run the Databricks ETL pipeline
Run the notebooks in `ETL/` in order on a Databricks workspace: `Bronze_layer.ipynb` → `Silver_layer.ipynb` → `Gold_layer.ipynb`.

## 🧠 Model Details

- **Algorithm:** `HistGradientBoostingRegressor` (scikit-learn's close cousin of XGBoost), selected after comparing against Ridge Regression and Random Forest, tuned via `RandomizedSearchCV`.
- **Target:** log-price (converted back to EGP for reporting).
- **Categorical encoding:** K-fold smoothed target encoding for high-cardinality Brand/Model; one-hot encoding for Transmission, Fuel Type, Source, Governorate.
- **Monotonic constraints:** price is constrained to never increase as car age or mileage increase, all else equal — prevents nonsensical predictions at the edges of the training distribution.
- **Uncertainty:** separate quantile models (10th / 50th / 90th percentile) give a price *range*, not just a point estimate.

**Performance:**
- R² ≈ 0.84
- MAE ≈ 270K EGP
- MAPE ≈ 20%
- 80% prediction interval empirical coverage validated on held-out test data

Top features by importance: **CarAge** (~48%), Brand, Model, Mileage. Full ranking in `feature_importance_v2.csv`.

## 📈 Power BI Dashboards

Interactive dashboards built on the Gold layer provide:
- Average price trend by year (market-wide and per-brand)
- Listings by location/governorate
- Fuel type share
- Top models by listing count, with avg/median price and mileage
- Brand-specific deep dives (e.g. Nissan, Toyota market analysis) with KPIs: total listings, average price, average mileage, most popular model

## ⚠️ Known Issues & Limitations

- **Current/future model-year collapse:** `CarAge` is clipped at a minimum of 0, so any car with a model year at or beyond the reference year gets `CarAge = 0`. The model currently cannot distinguish a car listed as this year's model from next year's model — they get an identical age signal, and since CarAge carries ~48% of feature importance, predictions for not-yet-released or brand-new model years are the least reliable segment. Planned fix: allow `CarAge` to go negative for future model years and retrain.
- **CarAge reference point:** training-time CarAge should ideally be computed from each listing's own `PostedOn` date rather than a single fixed reference year, so historical listings reflect their true age at time of sale instead of their age relative to today.
- **Unseen brands/models:** Brand and Model use smoothed target encoding, so a brand/model never seen in training falls back to the overall average price — expect lower accuracy for rare or exotic makes.
- **`ModelListingCount` at inference time:** can't be computed for a hypothetical car (there's no real listing to count), so a neutral default is used, which may slightly bias predictions for very rare or very common models.
- **Missing engine data for Hatla2ee-sourced rows:** `EngineCC` is only available from ContactCars; Hatla2ee rows fall back to a `HasEngineCC` flag + median imputation.

## 🔮 Future Work

- 🎯 Personalized car recommendation engine
- 📡 Real-time market monitoring
- 🤖 LLM integration for natural-language car search/advice
- 📱 Mobile application
- 🗣️ Arabic voice assistant
- 🔔 Automatic price-drop alerts

## 👥 Team

| Name | Role |
|---|---|
| Yusif Saeed | Data Engineering & Architecture |
| Mina Dawood | Data Engineering & Architecture |
| Abdalrahman Saleh | Data Engineering & Architecture |
| Mohamed Ateya Elhag | Data Engineering & Architecture |
| Khaled Abdelmageed | Data Engineering & Architecture |
| Karim Wessam | Data Engineering & Architecture |

**Supervised by:** Eng. Mohamed Hamed

---

**Made with ❤️ by the NHA Team**
