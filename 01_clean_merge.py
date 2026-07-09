"""
Clean and merge ContactCars + Hatla2ee datasets into one unified dataset
ready for price-prediction modeling.
"""
import pandas as pd
import numpy as np
import re

pd.set_option('display.max_columns', None)

# ---------- helpers ----------
def parse_number(x):
    """Turn '1,350,000 EGP' / '55,000 km' / '9,999,999 KM' into a float, else NaN."""
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if s.lower() in ('null', 'nan', ''):
        return np.nan
    s = re.sub(r'[^0-9.]', '', s)
    return float(s) if s else np.nan


# ---------- load ----------
cc = pd.read_csv('/mnt/user-data/uploads/Data_ContactCars.csv')
h = pd.read_csv('/mnt/user-data/uploads/hatla2ee.csv')

# ---------- clean ContactCars ----------
cc = cc.rename(columns={'Engine CC': 'EngineCC', 'Date Posted': 'PostedOn'})
cc['Price'] = cc['Price'].apply(parse_number)
cc['Mileage'] = cc['Mileage'].apply(parse_number)
cc['EngineCC'] = cc['EngineCC'].apply(parse_number)
cc['Year'] = pd.to_numeric(cc['Year'], errors='coerce')
cc['Transmission'] = cc['Transmission'].replace('Null', np.nan)
cc['FuelType'] = np.nan          # not collected by this source
cc['Location'] = np.nan          # not collected by this source
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
h['EngineCC'] = np.nan            # not collected by this source
h['Source'] = 'Hatla2ee'
h = h[['Brand', 'Model', 'Price', 'Year', 'Mileage', 'Transmission',
       'FuelType', 'EngineCC', 'Location', 'PostedOn', 'Source']]

# ---------- merge ----------
df = pd.concat([cc, h], ignore_index=True)

# normalize text fields
for col in ['Brand', 'Model', 'Transmission', 'FuelType', 'Location']:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col].isin(['nan', 'Null', '']), col] = np.nan

# ---------- sanity filtering ----------
before = len(df)
df = df.drop_duplicates()

# drop rows missing the essentials for a price model
df = df.dropna(subset=['Price', 'Year', 'Mileage', 'Brand', 'Model'])

# realistic bounds (Egyptian used-car market, 2026)
df = df[(df['Price'] >= 30_000) & (df['Price'] <= 20_000_000)]
df = df[(df['Year'] >= 1980) & (df['Year'] <= 2027)]
df = df[(df['Mileage'] >= 0) & (df['Mileage'] <= 500_000)]  # 9,999,999 placeholder etc. dropped

# EngineCC: real passenger/SUV engines run roughly 600-8000 cc; a couple of rows had
# 27,000 / 55,000 (clear data-entry errors) that were blowing up the linear model.
df.loc[(df['EngineCC'] < 600) | (df['EngineCC'] > 8000), 'EngineCC'] = np.nan

after = len(df)
print(f"Rows before cleaning: {before} | after cleaning: {after} | dropped: {before - after}")

# ---------- feature engineering ----------
df['CarAge'] = 2026 - df['Year']
df['CarAge'] = df['CarAge'].clip(lower=0)
df['LogPrice'] = np.log1p(df['Price'])

df.to_csv('/home/claude/car_price_model/cleaned_merged.csv', index=False)
print(df.shape)
print(df.head())
print(df.isnull().sum())
