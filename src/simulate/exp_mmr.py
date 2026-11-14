from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.recommend.content_based import ContentRecommender, load
from src.rerank.mmr import MMRReranker
from src.simulate.experiment import OUT_DIR, RandomRecommender, plot, run_all, summarize

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

    result = run_all(df, X, conditions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "day9_mmr.csv", index=False)
    summary = summarize(result)
    print("\n[후반 5라운드 평균]")
    print(summary.to_string())

    plot(result, Path("docs/figures/day9_mmr_curves.png"), "MMR re-ranking vs baselines")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(summary["utility"], summary["topic_entropy"])
    for name, row in summary.iterrows():
        ax.annotate(name, (row["utility"], row["topic_entropy"]),
                    textcoords="offset points", xytext=(5, 5))
    ax.set_xlabel("User utility (higher = more satisfied)")
    ax.set_ylabel("Topic entropy (higher = more diverse)")
    ax.set_title("Diversity vs Utility trade-off")
    fig.tight_layout()
    fig.savefig("docs/figures/day9_tradeoff.png", dpi=150)
    print("\n그래프 저장: docs/figures/day9_mmr_curves.png, day9_tradeoff.png")
