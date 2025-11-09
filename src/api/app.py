from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer

# ---------- Paths ----------
DATA_PATH = "data_processed/processed.csv"
MODEL_PATH = "data_processed/textcnn_model.pt"
VECTORIZER_PATH = "data_processed/vectorizer.pkl"
label_cols = ['egg', 'gluten', 'milk', 'peanuts', 'shellfish', 'soy', 'tree nuts']

# ---------- Load vectorizer ----------
vectorizer = joblib.load(VECTORIZER_PATH)
vocab_size = len(vectorizer.get_feature_names_out())

# ---------- Model ----------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear((vocab_size // 2) * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# ---------- Load trained model ----------
model = TextCNN(vocab_size, len(label_cols))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()
print("✅ CNN model loaded and API initialized!")

# ---------- Helper ----------
def encode(text: str):
    vec = vectorizer.transform([text.lower()]).toarray()[0]
    return torch.tensor(vec, dtype=torch.float32)

def predict_allergens(text: str):
    x = encode(text)
    with torch.no_grad():
        out = torch.sigmoid(model(x.unsqueeze(0)))[0]

    preds = {label: float(prob) for label, prob in zip(label_cols, out)}

    # 🧩 Custom thresholds — adjust sensitivity
    thresholds = {
        "milk": 0.55,
        "egg": 0.3,
        "gluten": 0.3,
        "peanuts": 0.3,
        "shellfish": 0.3,
        "soy": 0.3,
        "tree nuts": 0.3
    }

    detected = [k for k, v in preds.items() if v > thresholds[k]]
    detected = sorted(detected, key=lambda k: preds[k], reverse=True)

    return {"detected_allergens": detected, "confidence_scores": preds}

# ---------- FastAPI App ----------
app = FastAPI(title="Food Allergen Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # during local testing you can use "*"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class IngredientInput(BaseModel):
    ingredients: str

@app.get("/")
def root():
    return {"message": "Food Allergen Detection API is running!"}

@app.post("/predict")
def predict(input: IngredientInput):
    result = predict_allergens(input.ingredients)
    return {"input": input.ingredients, "result": result}
