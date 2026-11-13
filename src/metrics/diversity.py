from collections import Counter

import numpy as np

def normalized_entropy(labels, n_categories: int) -> float:
    """0(한 범주만) ~ 1(모든 범주 균등)"""
    counts = np.array(list(Counter(labels).values()), dtype=float)
    if len(counts) <= 1 or n_categories <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum() / np.log(n_categories))

def unique_count(labels) -> int:
    return len(set(labels))

def intra_list_diversity(X, idx: list[int]) -> float:
    """1 - 기사 간 평균 코사인 유사도. 높을수록 서로 다른 기사"""
    if len(idx) < 2:
        return 0.0
    sim = (X[idx] @ X[idx].T).toarray()
    n = len(idx)
    avg = (sim.sum() - np.trace(sim)) / (n * (n - 1))
    return float(1 - avg)

def report(df, X, idx: list[int]) -> dict:
    return {
        "topic_entropy": round(normalized_entropy(df.loc[idx, "topic"], df["topic"].nunique()), 3),
        "unique_sources": unique_count(df.loc[idx, "source"]),
        "ild": round(intra_list_diversity(X, idx), 3),
    }

if __name__ == "__main__":
    from src.recommend.content_based import ContentRecommender, load

    df, X = load()
    rec = ContentRecommender(X)
    random_feed = df.sample(10, random_state=1).index.tolist()
    history = df.sample(3, random_state=7).index.tolist()
    rec_feed = rec.recommend(history)
    print("[무작위 피드]   ", report(df, X, random_feed))
    print("[추천기 피드]   ", report(df, X, rec_feed))
