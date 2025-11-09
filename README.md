Here’s a clean and professional **`README.md`** file you can directly add to your GitHub repo 👇

---

```markdown
# 🧠 Food Allergen Detector using Deep Learning

A deep learning–based web application that identifies potential allergens from food ingredient lists.  
Built using **PyTorch**, **FastAPI**, and **scikit-learn**, this project uses a **TextCNN** model to analyze text and predict the presence of common allergens such as milk, egg, gluten, soy, peanuts, shellfish, and tree nuts.

---

## 🚀 Features
- ✅ Detects allergens from ingredient text using trained TextCNN model  
- 🧾 Real-time prediction API built with **FastAPI**  
- 💾 Offline dataset support (Open Food Facts processed data)  
- 📊 Model trained on cleaned and preprocessed food ingredient data  
- 🔥 Simple REST API endpoint for integration in web or mobile apps  

---

## 🧩 Tech Stack
| Category | Technology |
|-----------|-------------|
| **Language** | Python 3.10+ |
| **Framework** | FastAPI |
| **Deep Learning** | PyTorch |
| **ML Utilities** | scikit-learn, pandas, numpy |
| **Deployment** | Uvicorn (local) |
| **Model** | TextCNN for multi-label allergen detection |

---

## 🧱 Project Structure
```

food-allergen-detector/
│
├── data_raw/                   # Original dataset (offline copy)
├── data_processed/             # Cleaned and preprocessed data
├── src/
│   ├── data/
│   │   ├── ingest_off.py       # Loads offline dataset
│   ├── models/
│   │   ├── train_textcnn.py    # Trains TextCNN model
│   │   ├── evaluate_textcnn.py # Tests model with sample inputs
│   │   ├── api_fast.py         # FastAPI backend for predictions
│
├── requirements.txt            # Required Python packages
└── README.md                   # Project documentation

````

---

## ⚙️ Installation & Setup

### 1️⃣ Clone the repository
```bash
git clone https://github.com/aishwaryapvt/Food-allergen-detector.git
cd Food-allergen-detector
````

### 2️⃣ Create and activate virtual environment

```bash
python -m venv venv
venv\Scripts\activate      # (Windows)
# or
source venv/bin/activate   # (Mac/Linux)
```

### 3️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the model training (if needed)

```bash
python src/models/train_textcnn.py
```

### 5️⃣ Launch the FastAPI server

```bash
uvicorn src.models.api_fast:app --reload
```

Then open 👉 [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
to interact with the API using Swagger UI.

---

## 🧪 Example API Request

### **POST** `/predict`

**Body (JSON):**

```json
{
  "ingredients": "wheat flour, milk solids, egg, sugar"
}
```

**Response:**

```json
{
  "input": "wheat flour, milk solids, egg, sugar",
  "result": {
    "detected_allergens": ["milk", "egg", "gluten"],
    "confidence_scores": {
      "egg": 0.82,
      "gluten": 0.75,
      "milk": 0.90,
      "peanuts": 0.03,
      "shellfish": 0.01,
      "soy": 0.22,
      "tree nuts": 0.08
    }
  }
}
```

---

## 🧠 Model Overview

The **TextCNN** model uses convolutional layers over word embeddings to extract local features from ingredient text.
It’s trained as a **multi-label classifier** to predict probabilities for multiple allergens simultaneously.

---

## 📈 Future Enhancements

* 🌐 Integrate frontend for user-friendly interaction
* 🧬 Expand allergen categories using larger datasets
* ☁️ Deploy model to cloud (e.g., Render / Hugging Face Spaces)


