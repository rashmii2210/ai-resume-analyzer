"""
5_matcher.py — Compute a semantic match score between a resume and a job description.

Usage:
    python 5_matcher.py --resume path/to/resume.txt --job path/to/job_description.txt
    python 5_matcher.py --demo          # runs built-in sample resume/JD pairs
"""

import argparse
import re
import warnings

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")


# HELPER
def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def match_resume_to_job(resume_text: str, job_description: str, bert_model) -> dict:
    """
    Uses BERT to semantically compare resume vs job description.
    Returns a match score from 0 to 100.

    This is PURE ML — no rules, no hardcoding.
    BERT learned language from billions of sentences.
    Cosine similarity measures angle between two meaning-vectors.
    """
    clean_resume = clean_text(resume_text)
    clean_job = clean_text(job_description)

    resume_vector = bert_model.encode([clean_resume])
    job_vector = bert_model.encode([clean_job])

    raw_score = cosine_similarity(resume_vector, job_vector)[0][0]
    match_score = round(float(raw_score) * 100, 2)

    if match_score >= 75:
        grade, verdict = "Excellent Match", "Strong candidate for this role"
    elif match_score >= 60:
        grade, verdict = "Good Match", "Suitable candidate, minor gaps"
    elif match_score >= 45:
        grade, verdict = "Average Match", "Partial fit, needs improvement"
    elif match_score >= 30:
        grade, verdict = "Weak Match", "Significant gaps for this role"
    else:
        grade, verdict = "Poor Match", "Profile does not match this role"

    return {
        "match_score": match_score,
        "grade": grade,
        "verdict": verdict,
        "resume_length": len(clean_resume.split()),
        "job_length": len(clean_job.split()),
    }


def print_result(resume_label: str, jd_label: str, result: dict):
    print(f"\n  Resume  : {resume_label}")
    print(f"  Job     : {jd_label}")
    print(f"  Score   : {result['match_score']}%  →  {result['grade']}")
    print(f"  Verdict : {result['verdict']}")
    print(f"  {'-' * 50}")


# DEMO DATA (only used with --demo flag)
DEMO_JOBS = {
    "Data Scientist JD": """
        We are looking for a Data Scientist with experience in Python,
        Machine Learning, Deep Learning, NLP, TensorFlow, PyTorch,
        scikit-learn, SQL, MongoDB, BERT, Transformers, REST API,
        FastAPI, data analysis, statistical modeling, pandas, numpy.
        Strong knowledge of algorithms and model deployment required.
    """,
    "Finance JD": """
        We are looking for a Financial Analyst with experience in
        financial modeling, investment analysis, portfolio management,
        Excel, Bloomberg, equity research, risk management, budgeting,
        forecasting, financial statements, accounting, CFA certification.
    """,
    "Teacher JD": """
        We are looking for a Mathematics Teacher with experience in
        teaching algebra, calculus, statistics, curriculum development,
        lesson planning, classroom management, student assessment,
        interactive teaching methods, educational technology.
    """,
}

DEMO_RESUMES = {
    "Data Science Resume": """
        Python developer with expertise in machine learning and deep learning.
        Built NLP models using BERT and transformers. Experience with TensorFlow,
        scikit-learn, pandas, numpy. Developed REST APIs using FastAPI.
        Worked with MongoDB and SQL databases. Deployed models on AWS.
        Strong background in statistics, data analysis, and model evaluation.
        Bachelor of Technology in Computer Science. Published ML research paper.
    """,
    "Finance Resume": """
        Financial analyst with 4 years experience in investment banking.
        Proficient in financial modeling, valuation, Excel, Bloomberg.
        Experience in equity research, portfolio management, risk analysis.
        CFA Level 1 certified. Strong accounting and budgeting skills.
    """,
    "Unrelated Chef Resume": """
        Professional chef with 8 years experience in Italian and French cuisine.
        Expert in pastry making, menu design, kitchen management.
        Trained at culinary school. Experience in fine dining restaurants.
        Skills: cooking, baking, food presentation, kitchen hygiene.
    """,
}


def run_demo(bert_model):
    print("=" * 60)
    print("BERT SEMANTIC MATCHING — Resume vs Job Description")
    print("No rules. No hardcoding. Pure vector math.")
    print("=" * 60)

    pairs = [
        ("Data Science Resume", "Data Scientist JD"),
        ("Finance Resume", "Finance JD"),
        ("Finance Resume", "Data Scientist JD"),
        ("Unrelated Chef Resume", "Data Scientist JD"),
        ("Data Science Resume", "Teacher JD"),
        ("Finance Resume", "Teacher JD"),
    ]

    for resume_label, jd_label in pairs:
        result = match_resume_to_job(DEMO_RESUMES[resume_label], DEMO_JOBS[jd_label], bert_model)
        print_result(resume_label, jd_label, result)

    print("\n   BERT understands MEANING not just keywords.")
    print("   'machine learning' and 'ML models' score high together")
    print("   even though they use different words — that is semantic similarity.")


# CLI ENTRY POINT
def main():
    parser = argparse.ArgumentParser(description="Match a resume against a job description.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--resume", type=str, help="Path to a .txt file containing resume text")
    group.add_argument("--demo", action="store_true", help="Run built-in sample resume/JD pairs")
    parser.add_argument("--job", type=str, help="Path to a .txt file containing the job description (required with --resume)")
    args = parser.parse_args()

    if args.resume and not args.job:
        parser.error("--job is required when using --resume")

    print("Loading BERT model...")
    bert_model = SentenceTransformer("all-MiniLM-L6-v2")
    print("BERT loaded!\n")

    if args.demo:
        run_demo(bert_model)
    else:
        with open(args.resume, "r", encoding="utf-8") as f:
            resume_text = f.read()
        with open(args.job, "r", encoding="utf-8") as f:
            job_text = f.read()

        result = match_resume_to_job(resume_text, job_text, bert_model)
        print_result(args.resume, args.job, result)


if __name__ == "__main__":
    main()