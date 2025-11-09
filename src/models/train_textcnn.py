import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ---------------------------
# 1️⃣ Paths and data loading
# ---------------------------
DATA_PATH = "data_processed/processed.csv"
MODEL_PATH = "data_processed/textcnn_model.pt"

df = pd.read_csv(DATA_PATH)

# Ensure no NaN values in ingredients
df["clean_ingredients"] = df["clean_ingredients"].astype(str).fillna("")

# Label columns from your processed data
label_cols = ['egg', 'gluten', 'milk', 'peanuts', 'shellfish', 'soy', 'tree nuts']

# ---------------------------
# 2️⃣ Train-validation split
# ---------------------------
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df["clean_ingredients"], df[label_cols], test_size=0.2, random_state=42
)

# ---------------------------
# 3️⃣ Tokenization & vectorization
# ---------------------------
vectorizer = CountVectorizer(max_features=5000)
vectorizer.fit(df["clean_ingredients"])

def encode(text: str):
    """Convert text to a bag-of-words vector."""
    if not isinstance(text, str):
        text = ""
    vec = vectorizer.transform([text]).toarray()[0]
    return torch.tensor(vec, dtype=torch.float32)

# ---------------------------
# 4️⃣ PyTorch Dataset class
# ---------------------------
class FoodDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts.reset_index(drop=True)
        self.labels = labels.reset_index(drop=True).values

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        x = encode(self.texts.iloc[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.float32)
        return x, y

train_ds = FoodDataset(train_texts, train_labels)
val_ds = FoodDataset(val_texts, val_labels)
train_dl = DataLoader(train_ds, batch_size=8, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=8)

# ---------------------------
# 5️⃣ Define TextCNN model
# ---------------------------
class TextCNN(nn.Module):
    def __init__(self, vocab_size, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        # output size after pooling = vocab_size // 2
        self.fc = nn.Sequential(
            nn.Linear((vocab_size // 2) * 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, vocab_size]
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

vocab_size = len(vectorizer.get_feature_names_out())
model = TextCNN(vocab_size, len(label_cols))

# ---------------------------
# 6️⃣ Training setup
# ---------------------------
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

# ---------------------------
# 7️⃣ Training loop
# ---------------------------
for epoch in range(10):
    model.train()
    total_loss = 0
    for x, y in train_dl:
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/10 | Train Loss: {total_loss/len(train_dl):.4f}")

# ---------------------------
# 8️⃣ Save model and vectorizer
# ---------------------------
os.makedirs("data_processed", exist_ok=True)
torch.save(model.state_dict(), MODEL_PATH)

# Save vectorizer vocabulary for API
import joblib
joblib.dump(vectorizer, "data_processed/vectorizer.pkl")

print(f"✅ Model saved to {MODEL_PATH}")
print("✅ Vectorizer saved to data_processed/vectorizer.pkl")
