"""
Clean the new merged car dataset (data.csv) for price-prediction modeling.
Unlike the previous two-source version, this file already has Location,
Engine_CC, and Fuel_type for every row, plus a State (Used/New) column.
"""
import pandas as pd
import numpy as np

df = pd.read_csv('/mnt/user-data/uploads/data.csv')

# ---------- basic parsing ----------
df['Engine_CC'] = pd.to_numeric(df['Engine_CC'], errors='coerce')
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
df['Mileage'] = pd.to_numeric(df['Mileage'], errors='coerce')

for col in ['Brand', 'Model', 'Transmission', 'Fuel_type', 'Location', 'State']:
    df[col] = df[col].astype(str).str.strip()
    df.loc[df[col].isin(['nan', 'Null', '', 'None']), col] = np.nan

# ---------- extract City from Location ----------
# Location is either "Area, City" (e.g. "Nasr city, Cairo") or just "City" (e.g. "Cairo")
# or "Unknown". Take the part after the last comma when present.
def extract_city(loc):
    if pd.isna(loc) or loc == 'Unknown':
        return 'Unknown'
    parts = loc.split(',')
    return parts[-1].strip()

df['City'] = df['Location'].apply(extract_city)

# ---------- drop duplicates ----------
before = len(df)
df = df.drop_duplicates()

# ---------- drop rows missing essentials ----------
df = df.dropna(subset=['Price', 'Year', 'Mileage', 'Brand', 'Model'])

# ---------- realistic bounds ----------
df = df[(df['Price'] >= 30_000) & (df['Price'] <= 20_000_000)]
df = df[(df['Year'] >= 1980) & (df['Year'] <= 2027)]
df = df[(df['Mileage'] >= 0) & (df['Mileage'] <= 500_000)]  # drops 4M/999,999 placeholder rows

# Engine CC: real passenger/SUV engines run roughly 600-8000cc
df.loc[(df['Engine_CC'] < 600) | (df['Engine_CC'] > 8000), 'Engine_CC'] = np.nan

after = len(df)
print(f"Rows before cleaning: {before} | after cleaning: {after} | dropped: {before - after}")

# ---------- feature engineering ----------
df['CarAge'] = (2026 - df['Year']).clip(lower=0)
df['LogPrice'] = np.log1p(df['Price'])
df['IsNew'] = (df['State'] == 'New').astype(int)

df.to_csv('/home/claude/car_price_v2/cleaned.csv', index=False)
print(df.shape)
print(df[['Brand', 'Model', 'City', 'State', 'Price', 'CarAge', 'Mileage', 'Engine_CC']].head())
print(df.isnull().sum())
