# train_model.py
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer
import joblib

# Paths
CSV_PATH = "data/ai_mental_health_dataset.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "emotion_clf.joblib")
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs("data", exist_ok=True)

# If CSV doesn't exist, create a tiny demo dataset
if not os.path.exists(CSV_PATH):
    demo = [
        ("I feel so sad and lonely", "sad"),
        ("I'm really happy with how things went!", "happy"),
        ("I'm anxious about my exams", "anxious"),
        ("I feel hopeless", "sad"),
        ("This is exciting, I'm thrilled", "happy"),
        ("My heart is racing and I can't focus", "anxious"),
        ("I am angry about what happened", "angry"),
        ("I don't know what to do, everything is overwhelming", "anxious"),
        ("I feel great today, optimistic", "happy"),
        ("I can't stop crying", "sad"),
    ]
    df_demo = pd.DataFrame(demo, columns=["text", "label"])
    df_demo.to_csv(CSV_PATH, index=False)
    print(f"Demo dataset created at {CSV_PATH}")

# Load dataset
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["text", "label"])
texts = df["text"].astype(str).tolist()
labels = df["label"].astype(str).tolist()

# Encode labels to integers
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(labels)

# Embeddings
print("Loading embedding model:", EMBED_MODEL_NAME)
embedder = SentenceTransformer(EMBED_MODEL_NAME)
X = embedder.encode(texts, show_progress_bar=True)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Classifier
clf = LogisticRegression(max_iter=1000)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save classifier + label encoder + embedder name
joblib.dump({"clf": clf, "label_encoder": le, "embed_model_name": EMBED_MODEL_NAME}, MODEL_PATH)
print("Saved model to", MODEL_PATH)
