# src/models/evaluate_textcnn.py
import torch
import torch.nn as nn
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Config ----------
DATA_PATH = "data_processed/processed.csv"
MODEL_PATH = "data_processed/textcnn_model.pt"

label_cols = ['egg', 'gluten', 'milk', 'peanuts', 'shellfish', 'soy', 'tree nuts']

# ---------- Load data and vectorizer ----------
df = pd.read_csv(DATA_PATH)
df["clean_ingredients"] = df["clean_ingredients"].astype(str).fillna("")

vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(df["clean_ingredients"])

vocab_size = len(vectorizer.get_feature_names_out())

# ---------- Define same TextCNN model ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
        )
        self.fc = nn.Sequential(
            nn.Linear((vocab_size // 2) * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- Load trained model ----------
model = TextCNN(vocab_size, len(label_cols))
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
print("✅ Model loaded successfully!")

# ---------- Helper: encode text ----------
def encode(text: str):
    vec = vectorizer.transform([text.lower()]).toarray()[0]
    return torch.tensor(vec, dtype=torch.float32)

# ---------- Prediction function ----------
def predict_allergens(text):
    model.eval()
    x = encode(text)
    with torch.no_grad():
        out = torch.sigmoid(model(x.unsqueeze(0)))[0]
    preds = {label: float(prob) for label, prob in zip(label_cols, out)}
    detected = [k for k, v in preds.items() if v > 0.5]
    print("\n🧾 Input:", text)
    print("🔍 Predicted Allergens:", detected)
    print("Confidence Scores:")
    for k, v in preds.items():
        print(f"  {k:10s}: {v:.3f}")

# ---------- Test ----------
sample_ingredients = [
    "wheat flour, sugar, butter, milk solids, eggs",
    "roasted peanuts, salt, vegetable oil",
    "soy protein, oats, almond pieces",
    "shrimp, wheat noodles, garlic, butter, milk",
]

for text in sample_ingredients:
    predict_allergens(text)
