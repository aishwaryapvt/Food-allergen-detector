import pandas as pd
path = "ontology/allergen_ontology.csv"
df = pd.read_csv(path)
print(f"✅ Loaded ontology with {len(df)} entries")
print(df.head())

