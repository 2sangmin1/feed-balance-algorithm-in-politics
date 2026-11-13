from pathlib import Path

import joblib
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

DATA_DIR = Path("data/processed")

def build():
    df = pd.read_csv(DATA_DIR / "articles.csv")
    vectorizer = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True
    )
    X = vectorizer.fit_transform(df["title"])
    sparse.save_npz(DATA_DIR / "tfidf.npz", X)
    joblib.dump(vectorizer, DATA_DIR / "vectorizer.joblib")
    return df, X

def most_similar(df, X, idx: int, k: int = 5) -> pd.DataFrame:
    sims = linear_kernel(X[idx], X).ravel()
    sims[idx] = -1
    top = sims.argsort()[::-1][:k]
    return df.iloc[top][["title", "source", "topic"]].assign(score=sims[top].round(3))

if __name__ == "__main__":
    df, X = build()
    print(f"벡터 크기: {X.shape} (기사 수 x 특징 수)\n")
    for idx in df.sample(3, random_state=42).index:
        print(f"▶ 기준: {df.loc[idx, 'title']} ({df.loc[idx, 'source']})")
        print(most_similar(df, X, idx).to_string(index=False))
        print()
