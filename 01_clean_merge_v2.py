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


# brand acronyms that should always render upper-case regardless of which casing
# happened to be more frequent in the scraped data (e.g. "Gmc" -> "GMC")
BRAND_ACRONYMS = {'jac', 'byd', 'baic', 'ds', 'gmc', 'exeed', 'rox'}


def build_brand_canon_map(brand_series):
    """Collapse case-duplicate brands ('JAC' vs 'Jac') into one canonical spelling."""
    import collections
    groups = collections.defaultdict(list)
    for b in brand_series.dropna():
        groups[b.lower()].append(b)
    canon = {}
    for low, variants in groups.items():
        if low in BRAND_ACRONYMS:
            canon[low] = low.upper()
        else:
            canon[low] = collections.Counter(variants).most_common(1)[0][0]
    return canon


def fix_brand_model_pollution(df):
    """Some Hatla2ee rows have the FULL 'Brand Model' string dumped into Brand
    (e.g. Brand='Mercedes C 180', Model='Imported') because the scraper mis-split
    them. Detect these by: Brand has 2+ words AND the first word is itself a
    real standalone brand elsewhere in the data. Move the remainder into Model."""
    brand_lower = df['Brand'].dropna().astype(str).str.strip()
    single_word_brands = set(
        b.lower() for b in brand_lower.unique() if isinstance(b, str) and len(b.split()) == 1
    )
    n_words = brand_lower.str.split().str.len()
    first_word_lower = brand_lower.str.split().str[0].str.lower()
    polluted_notnull = (n_words >= 2) & first_word_lower.isin(single_word_brands)
    polluted = polluted_notnull.reindex(df.index, fill_value=False)

    fixed_brand = df['Brand'].copy()
    fixed_model = df['Model'].copy()
    words = brand_lower[polluted_notnull].str.split()
    fixed_brand.loc[polluted] = words.str[0].values
    fixed_model.loc[polluted] = words.str[1:].str.join(' ').values
    n_fixed = polluted.sum()
    if n_fixed:
        print(f"Fixed {n_fixed} rows where Brand contained 'Brand + Model' together")
    return fixed_brand, fixed_model


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

# NEW: drop rows where the whole row got column-shifted by the scraper - Brand ends up
# holding a price string like '630,000 EGP' and Model holds 'Brand Model Year' combined,
# with the remaining fields also unreliable. Found via ContactCars only, 173 rows.
# These can't be safely reconstructed (the embedded Year often doesn't even match the
# Year column), so they're dropped rather than guessed at.
row_shifted = df['Brand'].astype(str).str.contains('EGP', na=False)
n_shifted = row_shifted.sum()
if n_shifted:
    print(f"Dropping {n_shifted} rows where Brand/Model/Price got column-shifted by the scraper")
df = df[~row_shifted]

# NEW: fix rows where Brand and Model got mashed together (Hatla2ee scraper bug),
# e.g. Brand='Mercedes C 180' -> Brand='Mercedes', Model='C 180'
fixed_brand, fixed_model = fix_brand_model_pollution(df)
df['Brand'] = fixed_brand
df['Model'] = fixed_model

# NEW: collapse case-only brand duplicates (JAC/Jac, BYD/Byd, GMC/Gmc, ...)
brand_canon = build_brand_canon_map(df['Brand'])
df['Brand'] = df['Brand'].apply(lambda b: brand_canon.get(b.lower(), b) if pd.notna(b) else b)

# safety net: no legitimate brand name contains a digit; anything left is scraper junk
still_bad = df['Brand'].astype(str).str.contains(r'\d', regex=True, na=False)
if still_bad.sum():
    print(f"Dropping {still_bad.sum()} more rows with digit-containing Brand (unhandled corruption)")
df = df[~still_bad]

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
