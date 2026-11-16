from src.recommend.content_based import ContentRecommender, load
from src.rerank.mmr import MMRReranker
from src.simulate.experiment import RandomRecommender, run_and_report

LAMBDAS = [0.9, 0.7, 0.5, 0.3]

if __name__ == "__main__":
    df, X = load()
    base = ContentRecommender(X)
    conditions = {
        "content": lambda s: base,
        "random": lambda s: RandomRecommender(len(df), seed=1000 + s),
    }
    for lam in LAMBDAS:
        conditions[f"mmr_{lam}"] = lambda s, l=lam: MMRReranker(base, X, lam=l)
    run_and_report(df, X, conditions, "day10_mmr", "MMR re-ranking vs baselines")
