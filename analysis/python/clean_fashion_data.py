import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. File paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "FashionDataset.csv"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "fashion_products_clean.csv"


# --------------------------------------------------
# 2. Load raw data
# --------------------------------------------------

df = pd.read_csv(RAW_FILE)

print("Raw dataset shape:", df.shape)
print("\nRaw columns:")
print(df.columns.tolist())


# --------------------------------------------------
# 3. Remove unnecessary index column
# --------------------------------------------------

if "Unnamed: 0" in df.columns:
    df = df.drop(columns=["Unnamed: 0"])


# --------------------------------------------------
# 4. Rename columns
# --------------------------------------------------

df = df.rename(columns={
    "BrandName": "brand_name",
    "Deatils": "product_description",
    "Sizes": "sizes",
    "MRP": "mrp",
    "SellPrice": "selling_price",
    "Discount": "discount_pct",
    "Category": "category"
})


# --------------------------------------------------
# 5. Clean text columns
# --------------------------------------------------

text_columns = [
    "brand_name",
    "product_description",
    "sizes",
    "category"
]

for col in text_columns:
    df[col] = (
        df[col]
        .astype(str)
        .str.strip()
    )


# --------------------------------------------------
# 6. Clean MRP
# --------------------------------------------------

df["mrp"] = (
    df["mrp"]
    .astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
)

df["mrp"] = pd.to_numeric(
    df["mrp"],
    errors="coerce"
)


# --------------------------------------------------
# 7. Clean selling price
# --------------------------------------------------

df["selling_price"] = (
    df["selling_price"]
    .astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
)

df["selling_price"] = pd.to_numeric(
    df["selling_price"],
    errors="coerce"
)


# --------------------------------------------------
# 8. Clean discount
# --------------------------------------------------

df["discount_pct"] = (
    df["discount_pct"]
    .astype(str)
    .str.replace(r"[^\d.]", "", regex=True)
)

df["discount_pct"] = pd.to_numeric(
    df["discount_pct"],
    errors="coerce"
)


# --------------------------------------------------
# 9. Create calculated variables
# --------------------------------------------------

df["discount_calculated"] = (
    1 - (df["selling_price"] / df["mrp"])
) * 100


df["price_ratio"] = (
    df["selling_price"] / df["mrp"]
)


# --------------------------------------------------
# 10. Basic data validation
# --------------------------------------------------

df.loc[df["mrp"] <= 0, "mrp"] = pd.NA
df.loc[df["selling_price"] <= 0, "selling_price"] = pd.NA

df["discount_calculated"] = df["discount_calculated"].clip(
    lower=0,
    upper=100
)

df["price_ratio"] = df["price_ratio"].clip(
    lower=0,
    upper=1
)


# --------------------------------------------------
# 11. Create product ID
# --------------------------------------------------

df.insert(
    0,
    "product_id",
    range(1, len(df) + 1)
)


# --------------------------------------------------
# 12. Reorder columns
# --------------------------------------------------

df = df[
    [
        "product_id",
        "brand_name",
        "category",
        "product_description",
        "sizes",
        "mrp",
        "selling_price",
        "discount_pct",
        "discount_calculated",
        "price_ratio"
    ]
]


# --------------------------------------------------
# 13. Save cleaned dataset
# --------------------------------------------------

OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


# --------------------------------------------------
# 14. Validation report
# --------------------------------------------------

print("\n----------------------------------")
print("CLEANING COMPLETE")
print("----------------------------------")

print("Clean dataset shape:", df.shape)

print("\nMissing values:")
print(df.isna().sum())

print("\nUnique brands:", df["brand_name"].nunique())

print("\nUnique categories:", df["category"].nunique())

print("\nPrice summary:")
print(df[["mrp", "selling_price"]].describe())

print("\nSaved to:")
print(OUTPUT_FILE)
