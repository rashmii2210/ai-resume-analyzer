"""
4_predict.py — Predict job category from a resume.

Usage:
    python 4_predict.py --text "Software engineer with Python experience..."
    python 4_predict.py --pdf path/to/resume.pdf
    python 4_predict.py --demo          # runs built-in sample resumes
"""

import argparse
import pickle
import re
import warnings

import pdfplumber
import scipy.sparse as sp
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


# LOAD ALL SAVED MODELS
def load_models():
    print("Loading models...")
    with open("models/best_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/tfidf_vectorizer.pkl", "rb") as f:
        tfidf = pickle.load(f)
    with open("models/label_categories.pkl", "rb") as f:
        categories = pickle.load(f)
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("All models loaded!\n")
    return model, tfidf, categories, bert_model


# HELPER FUNCTIONS
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)    # remove URLs
    text = re.sub(r"\S+@\S+", "", text)    # remove emails
    text = re.sub(r"[^a-z\s]", " ", text)  # keep only letters
    text = re.sub(r"\s+", " ", text)        # clean spaces
    return text.strip()


def predict_from_text(raw_text: str, model, tfidf, categories, bert_model, label: str = "Resume"):
    """Predict job category from raw text string."""

    clean_full = clean_text(raw_text)
    clean_bert = clean_full[:3000]

    X_tfidf = tfidf.transform([clean_full])
    X_bert = bert_model.encode([clean_bert])
    X_bert_norm = normalize(X_bert)
    X_bert_sparse = sp.csr_matrix(X_bert_norm)
    X_combined = sp.hstack([X_tfidf, X_bert_sparse])

    predicted = model.predict(X_combined)[0]
    probs = model.predict_proba(X_combined)[0]

    scores = {
        categories[i]: round(float(probs[i]) * 100, 2)
        for i in range(len(categories))
    }
    scores = dict(sorted(scores.items(), key=lambda x: x[1], reverse=True))

    print("=" * 55)
    print(f"  {label}")
    print("=" * 55)
    print(f"  Predicted Category : {predicted}")
    print(f"  Confidence         : {scores[predicted]}%")
    print(f"\n  Top 5 Possible Roles:")
    for rank, (cat, score) in enumerate(list(scores.items())[:5], 1):
        bar = "█" * int(score / 4)
        print(f"    {rank}. {cat:<28} {score:>6}%  {bar}")
    print()

    return predicted, scores


def predict_from_pdf(pdf_path: str, model, tfidf, categories, bert_model, label: str = "Resume"):
    """Predict job category from a PDF file."""
    text = ""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"  Error reading PDF: {e}")
        return None, None

    if not text.strip():
        print("  Could not extract text from PDF")
        return None, None

    return predict_from_text(text, model, tfidf, categories, bert_model, label)


# DEMO SAMPLES (only used with --demo flag)
DEMO_RESUMES = {
    "Software Engineer Resume": """
        Software Engineer with 3 years of experience in Python, Java,
        and JavaScript. Developed REST APIs using Flask and Django.
        Worked with machine learning models using scikit-learn and TensorFlow.
        Experience with SQL databases, MongoDB, Git, Docker.
        Built data pipelines and deployed models on AWS.
        Strong knowledge of algorithms, data structures, and system design.
        Bachelor of Technology in Computer Science.
        Skills: Python, Java, Machine Learning, Deep Learning, NLP,
        TensorFlow, SQL, MongoDB, Docker, AWS, Git, REST API.
    """,
    "Finance / Banking Resume": """
        Financial Analyst with 4 years of experience in investment banking
        and portfolio management. Proficient in financial modeling, valuation,
        budgeting, and forecasting. Experience with Excel, Bloomberg Terminal,
        and financial reporting. Knowledge of equity research, risk analysis,
        and capital markets. CFA Level 1 certified.
        Bachelor of Commerce in Finance and Accounting.
        Skills: Financial Modeling, Valuation, Excel, Bloomberg, Risk Analysis,
        Portfolio Management, Equity Research, Accounting, Budgeting.
    """,
    "Healthcare / Nurse Resume": """
        Registered Nurse with 5 years of clinical experience in ICU and
        emergency care. Skilled in patient assessment, medication administration,
        and critical care procedures. Experience with electronic health records,
        medical equipment, and patient education. Strong knowledge of anatomy,
        pharmacology, and medical terminology.
        Bachelor of Science in Nursing. BLS and ACLS certified.
        Skills: Patient Care, Clinical Assessment, ICU, Emergency Care,
        Medical Records, Medication, Pharmacology, Critical Care.
    """,
    "Teacher Resume": """
        High School Mathematics Teacher with 6 years of teaching experience.
        Developed curriculum and lesson plans for algebra, calculus, and
        statistics. Used interactive teaching methods and digital tools to
        improve student engagement. Conducted parent-teacher meetings and
        student assessments. Strong classroom management skills.
        Bachelor of Education in Mathematics. State teaching certification.
        Skills: Mathematics, Curriculum Development, Lesson Planning,
        Classroom Management, Student Assessment, Calculus, Algebra.
    """,
}


def run_demo(model, tfidf, categories, bert_model):
    print("=" * 55)
    print("  DEMO MODE — Built-in sample resumes")
    print("=" * 55)
    print()
    for label, text in DEMO_RESUMES.items():
        predict_from_text(text, model, tfidf, categories, bert_model, label)


# CLI ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="Predict job category from a resume.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Raw resume text to classify")
    group.add_argument("--pdf", type=str, help="Path to a resume PDF file")
    group.add_argument("--demo", action="store_true", help="Run built-in sample resumes")
    args = parser.parse_args()

    model, tfidf, categories, bert_model = load_models()

    if args.demo:
        run_demo(model, tfidf, categories, bert_model)
    elif args.pdf:
        predict_from_pdf(args.pdf, model, tfidf, categories, bert_model, label=args.pdf)
    elif args.text:
        predict_from_text(args.text, model, tfidf, categories, bert_model, label="Resume")


if __name__ == "__main__":
    main()