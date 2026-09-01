# FUTURE_ML_03 - Resume / Candidate Screening System

## 📌 Project Overview
This project builds an ML system that automatically screens, scores, and ranks resumes based on a given job description. This helps recruiters shortlist candidates faster and identify skill gaps.

## 📊 Dataset Used
- **Source:** Resume Dataset (Kaggle)
- **Total Resumes:** 2,484
- **Categories:** 24

## 🛠️ Tools & Libraries
- Python
- Pandas, NumPy
- NLTK (text preprocessing)
- spaCy (skill extraction)
- Scikit-learn (TF-IDF, cosine similarity)
- Matplotlib (visualization)

## 📈 How It Works
1. **Text Cleaning:** Lowercasing, punctuation removal, URL removal
2. **Skill Extraction:** spaCy PhraseMatcher matches skills from a curated list
3. **Feature Extraction:** TF-IDF vectorization (1-2 grams)
4. **Similarity Scoring:** Cosine similarity between resume and job description
5. **Ranking:** Candidates sorted by fit score (highest to lowest)
6. **Skill Gap Analysis:** Missing skills identified for each candidate

## 📊 Example: Data Analyst Role

### Top Candidate
- **Category:** DIGITAL-MEDIA
- **Fit Score:** 0.316
- **Matched Skills:** dashboard, excel, power bi, python, reporting, sql, tableau
- **Missing Skills:** communication, data analysis, data visualization, machine learning, numpy, pandas, statistics

### Required Skills Detected
communication, dashboard, data analysis, data visualization, excel, machine learning, numpy, pandas, power bi, python, reporting, sql, statistics, tableau

### Most Common Missing Skills
numpy (9), power bi (9), pandas (8), data visualization (7), machine learning (7), statistics (7), dashboard (7)

## 📂 Files in Repository
- `resume_screener.py` - Main Python script
- `Resume/Resume.csv` - Dataset
- `ranked_candidates.csv` - Top candidates with scores
- `top_candidates_fit_scores.png` - Visual comparison
- `skill_coverage.png` - Required skill coverage
- `screening_summary.txt` - Business insights
- `README.md` - This file
- `requirements.txt` - Dependencies

## 🚀 How to Run
1. Clone this repository
2. Install dependencies: `pip install -r requirements.txt`
3. Download spaCy model: `python -m spacy download en_core_web_sm`
4. Run: `python resume_screener.py`

## 👤 Author
Rediet Girma
Machine Learning Intern - Future Interns"# FUTURE_ML_03" 
