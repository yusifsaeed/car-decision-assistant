"""
v2: Clean and merge ContactCars + Hatla2ee datasets.
Improvements over v1:
  - Extracts Governorate from the free-text Location field (was loaded but never used)
  - Per-brand IQR outlier filtering on Price instead of one global [30k, 20M] band
    (a global band lets in Hyundai Verna listed at 19M as much as it lets out a
    real Bentley at 25M -> per-brand bounds catch both)
  - Keeps a raw listing count per Brand/Model so rare/exotic models can be flagged
"""
import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', None)

def parse_number(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in ('null', 'nan', ''):
        return np.nan
    s = re.sub(r'[^0-9.]', '', s)
    return float(s) if s else np.nan


def extract_governorate(loc):
    """'Tagamo3 - New Cairo, Cairo' -> 'Cairo'; 'Giza' -> 'Giza'; NaN -> NaN"""
    if pd.isna(loc):
        return np.nan
    parts = str(loc).split(',')
    return parts[-1].strip()


# ---------- load ----------
cc = pd.read_csv('/home/claude/car_price_model/Data_ContactCars.csv')
h = pd.read_csv('/home/claude/car_price_model/hatla2ee.csv')

# ---------- clean ContactCars ----------
cc = cc.rename(columns={'Engine CC': 'EngineCC', 'Date Posted': 'PostedOn'})
cc['Price'] = cc['Price'].apply(parse_number)
cc['Mileage'] = cc['Mileage'].apply(parse_number)
cc['EngineCC'] = cc['EngineCC'].apply(parse_number)
cc['Year'] = pd.to_numeric(cc['Year'], errors='coerce')
cc['Transmission'] = cc['Transmission'].replace('Null', np.nan)
cc['FuelType'] = np.nan
cc['Location'] = np.nan
cc['Source'] = 'ContactCars'
cc = cc[['Brand', 'Model', 'Price', 'Year', 'Mileage', 'Transmission',
         'FuelType', 'EngineCC', 'Location', 'PostedOn', 'Source']]

# ---------- clean Hatla2ee ----------
h = h.rename(columns={'Transmission type': 'Transmission', 'Fuel type': 'FuelType',
                       'Posted On': 'PostedOn'})
h['Price'] = h['Price'].apply(parse_number)
h['Mileage'] = h['Mileage'].apply(parse_number)
h['Year'] = h['Year'].replace('Null', np.nan)
h['Year'] = pd.to_numeric(h['Year'], errors='coerce')
h['Transmission'] = h['Transmission'].replace('Null', np.nan)
h['FuelType'] = h['FuelType'].replace('Null', np.nan)
h['EngineCC'] = np.nan
h['Source'] = 'Hatla2ee'
h = h[['Brand', 'Model', 'Price', 'Year', 'Mileage', 'Transmission',
       'FuelType', 'EngineCC', 'Location', 'PostedOn', 'Source']]

# ---------- merge ----------
df = pd.concat([cc, h], ignore_index=True)

for col in ['Brand', 'Model', 'Transmission', 'FuelType', 'Location']:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col].isin(['nan', 'Null', '']), col] = np.nan

# NEW: governorate feature (low-cardinality, usable as a categorical)
df['Governorate'] = df['Location'].apply(extract_governorate)

before = len(df)
df = df.drop_duplicates()
df = df.dropna(subset=['Price', 'Year', 'Mileage', 'Brand', 'Model'])

# basic sanity bounds first (kills placeholder junk like 9,999,999 km)
df = df[(df['Price'] >= 30_000) & (df['Price'] <= 30_000_000)]
df = df[(df['Year'] >= 1980) & (df['Year'] <= 2027)]
df = df[(df['Mileage'] >= 0) & (df['Mileage'] <= 500_000)]
df.loc[(df['EngineCC'] < 600) | (df['EngineCC'] > 8000), 'EngineCC'] = np.nan

# NEW: per-brand IQR outlier filtering on Price (catches within-brand data-entry errors
# that a single global band misses, e.g. a stray Hyundai at 19M or a Bentley at 60k)
def brand_iqr_mask(s, k=3.0):
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    if iqr == 0:
        return pd.Series(True, index=s.index)
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return s.between(lo, hi)

keep_mask = df.groupby('Brand')['Price'].transform(lambda s: brand_iqr_mask(s))
df = df[keep_mask]

# NEW: listing count per Brand+Model, useful as a "how common/well-known is this exact
# trim" signal, and to flag rare models the encoder has little data on
model_counts = df.groupby(['Brand', 'Model']).size().rename('ModelListingCount')
df = df.merge(model_counts, on=['Brand', 'Model'], how='left')

after = len(df)
print(f"Rows before cleaning: {before} | after cleaning: {after} | dropped: {before - after}")

df['CarAge'] = 2026 - df['Year']
df['CarAge'] = df['CarAge'].clip(lower=0)
# NEW: usage-intensity feature - km driven per year of ownership (capped to avoid
# div-by-zero blowups on brand-new cars)
df['MileagePerYear'] = df['Mileage'] / df['CarAge'].replace(0, 1)
df['MileagePerYear'] = df['MileagePerYear'].clip(upper=100_000)
df['LogPrice'] = np.log1p(df['Price'])

df.to_csv('/home/claude/car_price_model/cleaned_merged_v2.csv', index=False)
print(df.shape)
print(df[['Brand', 'Model', 'Governorate', 'ModelListingCount', 'MileagePerYear']].head())
print(df.isnull().sum())
