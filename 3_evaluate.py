import pandas as pd
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import scipy.sparse as sp

# ── Load data ──
df = pd.read_csv("data/Resume_clean.csv")
category_counts = df['Category'].value_counts()
valid_categories = category_counts[category_counts >= 40].index
df = df[df['Category'].isin(valid_categories)].reset_index(drop=True)

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

df['clean_full'] = df['Resume_str'].apply(clean_text)

# ── Load saved artifacts ──
print("Loading saved TF-IDF and BERT embeddings...")
with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

X_bert  = np.load("models/bert_embeddings.npy")
X_tfidf = tfidf.transform(df['clean_full'])

# Must combine features the same way as during training
X_bert_norm   = normalize(X_bert)
X_bert_sparse = sp.csr_matrix(X_bert_norm)
X_combined    = sp.hstack([X_tfidf, X_bert_sparse])

y = df['Category']

X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.20, random_state=42, stratify=y
)

y_pred = model.predict(X_test)

# ── Overall accuracy ──
print("=" * 50)
print("OVERALL ACCURACY")
print("=" * 50)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy : {round(acc * 100, 2)}%")
print(f"Correct  : {sum(y_pred == y_test)} / {len(y_test)}")
print(f"Wrong    : {sum(y_pred != y_test)} / {len(y_test)}")

# ── Per-category report ──
print("\n" + "=" * 50)
print("PER CATEGORY REPORT")
print("=" * 50)
print(classification_report(y_test, y_pred, zero_division=0))

# ── Confusion matrix ──
categories = sorted(df['Category'].unique())
cm = confusion_matrix(y_test, y_pred, labels=categories)

plt.figure(figsize=(16, 12))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=categories, yticklabels=categories)
plt.title('Confusion Matrix — TF-IDF + BERT + Logistic Regression', fontsize=14)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("models/confusion_matrix_final.png", dpi=150)
plt.show()
print("Saved to models/confusion_matrix_final.png")

# ── Best/worst performing categories ──
report     = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
cat_scores = {k: round(v['f1-score']*100, 1) for k, v in report.items() if k in categories}
sorted_scores = sorted(cat_scores.items(), key=lambda x: x[1], reverse=True)

print("\n" + "=" * 50)
print("BEST PREDICTED CATEGORIES")
print("=" * 50)
for cat, score in sorted_scores[:5]:
    print(f"  {cat:<25} F1: {score}%")

print("\n" + "=" * 50)
print("WORST PREDICTED CATEGORIES")
print("=" * 50)
for cat, score in sorted_scores[-5:]:
    print(f"  {cat:<25} F1: {score}%")