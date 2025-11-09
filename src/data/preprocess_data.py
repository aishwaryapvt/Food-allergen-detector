# src/data/preprocess_data.py
import os
import pandas as pd
import re

RAW_PATH = "data_raw/off_subset.csv"
ONTOLOGY_PATH = "ontology/allergen_ontology.csv"
PROCESSED_PATH = "data_processed/processed.csv"

os.makedirs("data_processed", exist_ok=True)

# Load datasets
df = pd.read_csv(RAW_PATH)
ontology = pd.read_csv(ONTOLOGY_PATH)

print(f"✅ Loaded {len(df)} food products")
print(f"✅ Loaded {len(ontology)} ontology mappings")

# Normalize ingredient text
def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r"\d+g|\d+mg|\d+%", "", text)
    text = re.sub(r"[^a-zA-Z, ]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

df["clean_ingredients"] = df["ingredients_text"].apply(clean_text)

# Map allergens using ontology
def map_ontology(ingredients):
    found = []
    for _, row in ontology.iterrows():
        if row["synonym"] in ingredients:
            found.append(row["canonical_allergen"])
    return list(set(found))

df["detected_allergens"] = df["clean_ingredients"].apply(map_ontology)

# Combine with labeled allergens
df["true_allergens"] = df["allergens"].fillna("").apply(lambda x: [a.strip() for a in x.split(",") if a.strip()])

# Create all unique allergen labels
all_labels = sorted(set(sum(df["true_allergens"], [])))
print(f"🧾 Found {len(all_labels)} allergen classes: {all_labels}")

# Multi-hot encode allergens
for label in all_labels:
    df[label] = df["true_allergens"].apply(lambda x: 1 if label in x else 0)

# Save processed dataset
df.to_csv(PROCESSED_PATH, index=False)
print(f"✅ Saved processed dataset to {PROCESSED_PATH}")
print(df.head())
