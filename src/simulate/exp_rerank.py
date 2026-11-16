from src.recommend.content_based import ContentRecommender, load
from src.rerank.calibration import CalibratedReranker
from src.rerank.mmr import MMRReranker
from src.simulate.experiment import RandomRecommender, run_and_report

if __name__ == "__main__":
    df, X = load()
    base = ContentRecommender(X)
    topics = df["topic"].values
    conditions = {
        "content": lambda s: base,
        "random": lambda s: RandomRecommender(len(df), seed=1000 + s),
    }
    for lam in [0.7, 0.5]:
        conditions[f"mmr_{lam}"] = lambda s, l=lam: MMRReranker(base, X, lam=l)
    for lam in [0.3, 0.6, 0.9]:
        conditions[f"calib_{lam}"] = lambda s, l=lam: CalibratedReranker(base, topics, lam=l)
    run_and_report(df, X, conditions, "day11_rerank", "MMR vs Calibration")
