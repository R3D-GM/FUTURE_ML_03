import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import spacy
from spacy.matcher import PhraseMatcher

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

for pkg in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.download(pkg, quiet=True)
    except Exception:
        pass

STOPWORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

nlp = spacy.load("en_core_web_sm")

DATA_PATH = "Resume/Resume.csv"
TEXT_COL = "Resume_str"
CATEGORY_COL = "Category"
TOP_N = 10

JOB_TITLE = "Data Analyst"
JOB_DESCRIPTION = """
We are looking for a Data Analyst to join our team. The ideal candidate has
strong experience with Python, SQL, and Excel for data analysis. Experience
with data visualization tools such as Tableau or Power BI is required.
Familiarity with statistics, machine learning basics, and communication
skills to present findings to stakeholders is essential. Experience with
pandas, numpy, and dashboard reporting is a strong plus.
"""

SKILLS_LIST = [
    "python", "sql", "excel", "tableau", "power bi", "pandas", "numpy",
    "machine learning", "deep learning", "statistics", "data visualization",
    "communication", "java", "javascript", "aws", "azure", "docker",
    "kubernetes", "git", "scikit-learn", "tensorflow", "pytorch",
    "data analysis", "data cleaning", "reporting", "dashboard",
    "project management", "leadership", "presentation", "r programming",
    "spark", "hadoop", "etl", "power point", "word", "problem solving",
]

def nltk_clean(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s+#.]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def nltk_clean_and_lemmatize(text):
    tokens = word_tokenize(nltk_clean(text))
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
skill_patterns = [nlp.make_doc(skill) for skill in SKILLS_LIST]
matcher.add("SKILLS", skill_patterns)

def extract_skills(text):
    doc = nlp(nltk_clean(text))
    matches = matcher(doc)
    found = set()
    for match_id, start, end in matches:
        found.add(doc[start:end].text.lower())
    return sorted(found)

df = pd.read_csv(DATA_PATH, encoding="latin-1")
df.columns = [c.strip() for c in df.columns]
df = df.dropna(subset=[TEXT_COL]).reset_index(drop=True)
print(f"Loaded {len(df)} resumes")
print(f"Categories found: {df[CATEGORY_COL].nunique()} unique")

print("Extracting skills from resumes (this may take a minute)...")
df["extracted_skills"] = df[TEXT_COL].apply(extract_skills)
df["clean_text"] = df[TEXT_COL].apply(nltk_clean_and_lemmatize)

jd_skills = set(extract_skills(JOB_DESCRIPTION))
jd_clean = nltk_clean_and_lemmatize(JOB_DESCRIPTION)
print(f"\nJob: {JOB_TITLE}")
print(f"Required skills detected in JD: {sorted(jd_skills)}")

corpus = df["clean_text"].tolist() + [jd_clean]
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=1)
tfidf_matrix = vectorizer.fit_transform(corpus)

resume_vectors = tfidf_matrix[:-1]
jd_vector = tfidf_matrix[-1]

similarities = cosine_similarity(resume_vectors, jd_vector).flatten()
df["fit_score"] = similarities

def missing_skills(candidate_skills):
    return sorted(jd_skills - set(candidate_skills))

df["matched_skills"] = df["extracted_skills"].apply(lambda s: sorted(jd_skills & set(s)))
df["missing_skills"] = df["extracted_skills"].apply(missing_skills)

ranked = df.sort_values("fit_score", ascending=False).reset_index(drop=True)
top_candidates = ranked.head(TOP_N)

print(f"\n=== TOP {TOP_N} CANDIDATES FOR: {JOB_TITLE} ===\n")
for i, row in top_candidates.iterrows():
    print(f"#{i+1} | Fit Score: {row['fit_score']:.3f} | Category: {row.get(CATEGORY_COL, 'N/A')}")
    print(f"   Matched skills: {row['matched_skills']}")
    print(f"   Missing skills: {row['missing_skills']}")
    print()

output_cols = [CATEGORY_COL, "fit_score", "matched_skills", "missing_skills"]
top_candidates[output_cols].to_csv("ranked_candidates.csv", index=False)
print("Saved: ranked_candidates.csv")

plt.figure(figsize=(10, 6))
labels = [f"Candidate {i+1}" for i in range(len(top_candidates))]
plt.barh(labels[::-1], top_candidates["fit_score"].values[::-1], color="steelblue")
plt.xlabel("Fit Score (cosine similarity to job description)")
plt.title(f"Top {TOP_N} Candidates — {JOB_TITLE}")
plt.tight_layout()
plt.savefig("top_candidates_fit_scores.png")
plt.close()
print("Saved: top_candidates_fit_scores.png")

skill_counts = {skill: 0 for skill in jd_skills}
for skills in top_candidates["matched_skills"]:
    for s in skills:
        skill_counts[s] += 1

if skill_counts:
    plt.figure(figsize=(10, 5))
    skills_sorted = sorted(skill_counts.items(), key=lambda x: x[1], reverse=True)
    names, counts = zip(*skills_sorted)
    plt.bar(names, counts, color="darkorange")
    plt.ylabel(f"# of top {TOP_N} candidates with this skill")
    plt.title("Required Skill Coverage Among Top Candidates")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig("skill_coverage.png")
    plt.close()
    print("Saved: skill_coverage.png")

avg_score = top_candidates["fit_score"].mean()
most_common_missing = pd.Series(
    [s for skills in top_candidates["missing_skills"] for s in skills]
).value_counts()

summary = f"""
RESUME SCREENING SUMMARY — {JOB_TITLE}
------------------------------------------------
Total resumes screened: {len(df)}
Top {TOP_N} candidates average fit score: {avg_score:.3f}

How resumes are scored:
  Each resume and the job description are converted into TF-IDF vectors
  (word/phrase importance weighting), then compared using cosine similarity —
  a 0 to 1 score of how closely the resume's language overlaps with the JD's.
  Higher score = closer overall match to the role's language and requirements.

How skills are extracted:
  A curated skill vocabulary is matched against resume and JD text using
  spaCy's phrase matching, which finds exact skill mentions (e.g. "python",
  "power bi") regardless of surrounding sentence structure.

Why certain candidates rank higher:
  Candidates rank higher when their resume text more closely overlaps with the
  JD's terminology and when they mention more of the required skills explicitly.

Most common missing skills among top {TOP_N} candidates:
{most_common_missing.to_string() if not most_common_missing.empty else '  (none — top candidates cover all required skills)'}

What this means for a hiring team:
- The top-ranked list gives recruiters a fast starting shortlist instead of reading
  every resume manually.
- The missing-skills breakdown shows where candidates may need a follow-up question
  in an interview, or where the JD's required skills are hard to find in the pool.
- Fit score is a language-overlap signal, not a hard filter — recruiters should
  still review top candidates manually before rejecting anyone below the cutoff.
"""

print(summary)

with open("screening_summary.txt", "w") as f:
    f.write(summary)

print("\nAll files generated: ranked_candidates.csv, top_candidates_fit_scores.png, skill_coverage.png, screening_summary.txt")