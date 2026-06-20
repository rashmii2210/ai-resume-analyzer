from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import pickle
import io
import numpy as np
import re
import scipy.sparse as sp
import pdfplumber
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

app = FastAPI(title="AI Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Models load once at startup so each request only does inference, not loading
print("Loading models...")

with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/tfidf_vectorizer.pkl", "rb") as f:
    tfidf = pickle.load(f)

with open("models/label_categories.pkl", "rb") as f:
    categories = pickle.load(f)

bert_model = SentenceTransformer('all-MiniLM-L6-v2')

print("Models loaded successfully!\n")


def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def extract_text_from_upload(filename: str, content: bytes) -> str:
    """
    Extract text from an uploaded resume file.
    Supports .pdf (via pdfplumber) and plain text files (.txt and similar).
    """
    if filename.lower().endswith(".pdf"):
        text = ""
        with pdfplumber.open(io.BytesIO(content)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
        return text

    # Fall back to plain text decoding for .txt and similar files
    return content.decode("utf-8", errors="ignore")


def get_match_score(resume: str, job: str) -> float:
    r = bert_model.encode([resume])
    j = bert_model.encode([job])

    score = cosine_similarity(r, j)[0][0]
    return round(float(score) * 100, 2)


def predict_role(resume_text: str) -> str:
    clean = clean_text(resume_text)

    X_tfidf = tfidf.transform([clean])

    X_bert = bert_model.encode([clean])
    X_bert = normalize(X_bert)
    X_bert_sparse = sp.csr_matrix(X_bert)

    X = sp.hstack([X_tfidf, X_bert_sparse])

    pred = model.predict(X)[0]
    return pred


@app.get("/")
def home():
    return {"message": "AI Resume Analyzer API is running!"}


@app.post("/analyze")
async def analyze_resume(
    resume: UploadFile,
    job_description: str = Form(...)
):
    try:
        content = await resume.read()

        if not content:
            raise HTTPException(status_code=400, detail="Empty resume file")

        resume_text = extract_text_from_upload(resume.filename, content)

        if not resume_text.strip():
            raise HTTPException(
                status_code=400,
                detail="Could not extract any text from the uploaded file."
            )

        clean_resume = clean_text(resume_text)
        clean_job = clean_text(job_description)

        match_score = get_match_score(clean_resume, clean_job)
        role = predict_role(clean_resume)

        return {
            "predicted_role": role,
            "match_score": match_score
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)