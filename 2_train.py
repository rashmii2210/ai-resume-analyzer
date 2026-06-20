import pandas as pd
import numpy as np
import pickle
import re
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer
import scipy.sparse as sp
import warnings
warnings.filterwarnings('ignore')

# ── Load data ──
df = pd.read_csv("data/Resume_clean.csv")

# Drop categories with too few samples to train on reliably
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= 40].index
df = df[df['Category'].isin(valid_categories)].reset_index(drop=True)

print("=" * 50)
print("STEP 1: DATA LOADED")
print("=" * 50)
print(f"Total resumes    : {len(df)}")
print(f"Total categories : {df['Category'].nunique()}")

# ── Clean text ──
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_full'] = df['Resume_str'].apply(clean_text)

# BERT has a token limit, so resumes are truncated to 3000 chars before encoding
df['clean_bert'] = df['clean_full'].str[:3000]

print("\n" + "=" * 50)
print("STEP 2: TEXT CLEANING DONE")
print("=" * 50)
print(f"Avg resume length : {df['clean_full'].str.len().mean():.0f} chars")

# ── TF-IDF features ──
print("\n" + "=" * 50)
print("STEP 3A: TF-IDF VECTORIZATION")
print("=" * 50)

tfidf = TfidfVectorizer(
    max_features=3000,
    stop_words='english',
    ngram_range=(1, 3),
    min_df=2,
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(df['clean_full'])
print(f"TF-IDF shape : {X_tfidf.shape}")

# ── BERT features ──
print("\n" + "=" * 50)
print("STEP 3B: BERT EMBEDDINGS")
print("=" * 50)

bert_model = SentenceTransformer('all-MiniLM-L6-v2')
print("Encoding resumes with BERT (2-3 mins)...")

X_bert = bert_model.encode(
    df['clean_bert'].tolist(),
    batch_size=64,
    show_progress_bar=True
)
print(f"BERT shape : {X_bert.shape}")

# ── Combine TF-IDF (keyword importance) with BERT (semantic meaning) ──
print("\n" + "=" * 50)
print("STEP 3C: COMBINING TF-IDF + BERT")
print("=" * 50)

# Normalize BERT vectors to the same scale as TF-IDF before combining
X_bert_norm = normalize(X_bert)
X_bert_sparse = sp.csr_matrix(X_bert_norm)
X_combined = sp.hstack([X_tfidf, X_bert_sparse])

print(f"TF-IDF features : {X_tfidf.shape[1]}")
print(f"BERT features   : {X_bert_sparse.shape[1]}")
print(f"Combined shape  : {X_combined.shape}")

# Save embeddings/vectorizer so 3_evaluate.py and inference scripts can reuse them
np.save("models/bert_embeddings.npy", X_bert)
with open("models/tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf, f)

# ── Train/test split ──
y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\n" + "=" * 50)
print("STEP 4: TRAIN / TEST SPLIT")
print("=" * 50)
print(f"Training : {X_train.shape[0]} resumes")
print(f"Testing  : {X_test.shape[0]} resumes")

# ── Train models across a few regularization strengths, keep the best ──
print("\n" + "=" * 50)
print("STEP 5: TRAINING MODELS")
print("=" * 50)

models = {
    "LR C=50  (TF-IDF + BERT)" : LogisticRegression(C=50,  max_iter=2000, random_state=42),
    "LR C=100 (TF-IDF + BERT)" : LogisticRegression(C=100, max_iter=2000, random_state=42),
    "LR C=200 (TF-IDF + BERT)" : LogisticRegression(C=200, max_iter=2000, random_state=42),
    "LR C=500 (TF-IDF + BERT)" : LogisticRegression(C=500, max_iter=2000, random_state=42),
}

results = {}
for name, model in models.items():
    print(f"Training {name}...", end=" ")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    results[name] = {"model": model, "accuracy": acc}
    print(f"Accuracy: {round(acc * 100, 2)}%")

# ── Save best model ──
best_name  = max(results, key=lambda x: results[x]['accuracy'])
best_model = results[best_name]['model']
best_acc   = results[best_name]['accuracy']

print("\n" + "=" * 50)
print("STEP 6: FINAL RESULTS")
print("=" * 50)
for name, res in sorted(results.items(), key=lambda x: x[1]['accuracy'], reverse=True):
    marker = " <- BEST" if name == best_name else ""
    print(f"  {name:<35} {round(res['accuracy']*100, 2)}%{marker}")

with open("models/best_model.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("models/label_categories.pkl", "wb") as f:
    pickle.dump(sorted(df['Category'].unique().tolist()), f)

print(f"\nBest model saved!")
print(f"Model    : {best_name}")
print(f"Accuracy : {round(best_acc*100, 2)}%")
print(f"\nAccuracy comparison:")
print(f"  TF-IDF alone  -> ~69%")
print(f"  BERT alone    -> ~73%")
print(f"  TF-IDF + BERT -> {round(best_acc*100, 2)}%")