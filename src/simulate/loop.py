from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.diversity import intra_list_diversity, normalized_entropy
from src.recommend.content_based import ContentRecommender, load

ROUNDS = 30
K = 10
BIAS = 0.3        # 선호 주제에 대한 '살짝' 더 높은 관심
TEMPERATURE = 0.1 # 낮을수록 점수 높은 기사를 더 확실히 클릭

def simulate(df, X, rec, pref_topic: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = df["topic"].values
    n_topics = df["topic"].nunique()
    history = rng.choice(len(df), size=5, replace=False).tolist()

    log = []
    for r in range(1, ROUNDS + 1):
        feed = rec.recommend(history, k=K)
        utility = rec.scores(history)[feed] + BIAS * (topics[feed] == pref_topic)
        logits = utility / TEMPERATURE
        p = np.exp(logits - logits.max())
        p /= p.sum()
        history.append(int(rng.choice(feed, p=p)))
        log.append({
            "round": r,
            "topic_entropy": normalized_entropy(topics[feed], n_topics),
            "pref_share": float((topics[feed] == pref_topic).mean()),
            "ild": intra_list_diversity(X, feed),
        })
    return pd.DataFrame(log)

if __name__ == "__main__":
    df, X = load()
    rec = ContentRecommender(X)
    runs = []
    for topic in df["topic"].unique():
        for seed in range(3):
            runs.append(simulate(df, X, rec, topic, seed).assign(user=f"{topic}-{seed}"))
    result = pd.concat(runs)
    mean = result.groupby("round")[["topic_entropy", "pref_share", "ild"]].mean()

    print(f"가상 유저 {result['user'].nunique()}명, {ROUNDS}라운드")
    print("\n[초반 5라운드 평균]")
    print(mean.head(5).mean().round(3).to_string())
    print("\n[후반 5라운드 평균]")
    print(mean.tail(5).mean().round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col, title in zip(
        axes,
        ["topic_entropy", "pref_share", "ild"],
        ["Topic entropy (higher = diverse)", "Share of preferred topic", "Intra-list diversity"],
    ):
        ax.plot(mean.index, mean[col])
        ax.set_title(title)
        ax.set_xlabel("Round")
    fig.suptitle("Feedback loop: small preference -> narrowing feed")
    fig.tight_layout()
    out = Path("docs/figures/day7_feedback_loop.png")
    fig.savefig(out, dpi=150)
    print(f"\n그래프 저장: {out}")
