from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

DATA_DIR = Path("data/processed")

def load():
    df = pd.read_csv(DATA_DIR / "articles.csv")
    X = sparse.load_npz(DATA_DIR / "tfidf.npz")
    return df, X

class ContentRecommender:
    """클릭 기록의 평균 벡터(사용자 프로필)와 유사한 기사를 추천"""

    def __init__(self, X):
        self.X = X

    def profile(self, history: list[int]) -> np.ndarray:
        v = np.asarray(self.X[history].mean(axis=0)).ravel()
        norm = np.linalg.norm(v)
        return v / norm if norm > 0 else v

    def scores(self, history: list[int]) -> np.ndarray:
        return np.asarray(self.X @ self.profile(history)).ravel()

    def recommend(self, history: list[int], k: int = 10) -> list[int]:
        s = self.scores(history)
        s[list(set(history))] = -np.inf
        top = np.argpartition(-s, k)[:k]
        return top[np.argsort(-s[top])].tolist()

if __name__ == "__main__":
    df, X = load()
    rec = ContentRecommender(X)
    history = df.sample(3, random_state=7).index.tolist()
    print("[클릭 기록]")
    print(df.loc[history, ["title", "topic"]].to_string(index=False))
    print("\n[추천 결과 Top 10]")
    print(df.loc[rec.recommend(history), ["title", "topic", "source"]].to_string(index=False))
