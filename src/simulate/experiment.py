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
N_INIT = 5
SEEDS = 10
BIAS = 0.3
TEMPERATURE = 0.1
OUT_DIR = Path("docs/results")
METRICS = ["topic_entropy", "pref_share", "ild", "utility"]
TITLES = {
    "topic_entropy": "Topic entropy (higher = diverse)",
    "pref_share": "Share of preferred topic",
    "ild": "Intra-list diversity",
    "utility": "User utility of feed",
}

class RandomRecommender:
    def __init__(self, n_items: int, seed: int = 0):
        self.n = n_items
        self.rng = np.random.default_rng(seed)

    def recommend(self, history: list[int], k: int = K) -> list[int]:
        candidates = np.setdiff1d(np.arange(self.n), history)
        return self.rng.choice(candidates, size=k, replace=False).tolist()

class SimUser:
    """유저의 실제 취향: 지금까지 본 기사와의 유사도 + 선호 주제 보너스"""

    def __init__(self, X, topics: np.ndarray, pref_topic: str):
        self.taste = ContentRecommender(X)  # 프로필 계산 로직만 재사용
        self.topics = topics
        self.pref = pref_topic

    def utility(self, history: list[int], items: list[int]) -> np.ndarray:
        items = np.asarray(items)
        return self.taste.scores(history)[items] + BIAS * (self.topics[items] == self.pref)

def simulate(df, X, rec, pref_topic: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    topics = df["topic"].values
    n_topics = df["topic"].nunique()
    user = SimUser(X, topics, pref_topic)
    history = rng.choice(len(df), size=N_INIT, replace=False).tolist()

    log = []
    for r in range(1, ROUNDS + 1):
        feed = rec.recommend(history, k=K)
        u = user.utility(history, feed)
        p = np.exp((u - u.max()) / TEMPERATURE)
        p /= p.sum()
        log.append({
            "round": r,
            "topic_entropy": normalized_entropy(topics[feed], n_topics),
            "pref_share": float((topics[feed] == pref_topic).mean()),
            "ild": intra_list_diversity(X, feed),
            "utility": float(u.mean()),
        })
        history.append(int(rng.choice(feed, p=p)))
    return pd.DataFrame(log)

def run_all(df, X, conditions: dict) -> pd.DataFrame:
    """conditions: {조건 이름: seed를 받아 추천기를 돌려주는 함수}"""
    runs = []
    for name, make_rec in conditions.items():
        for topic in df["topic"].unique():
            for seed in range(SEEDS):
                res = simulate(df, X, make_rec(seed), topic, seed)
                runs.append(res.assign(condition=name, user=f"{topic}-{seed}"))
        print(f"  {name} 완료")
    return pd.concat(runs, ignore_index=True)

def summarize(result: pd.DataFrame, last: int = 5) -> pd.DataFrame:
    tail = result[result["round"] > ROUNDS - last]
    return tail.groupby("condition")[METRICS].mean().round(3)

def plot(result: pd.DataFrame, out_path: Path, title: str):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 4))
    for ax, col in zip(axes, METRICS):
        g = result.groupby(["condition", "round"])[col].agg(["mean", "std", "count"]).reset_index()
        for name, sub in g.groupby("condition"):
            ci = 1.96 * sub["std"] / np.sqrt(sub["count"])
            ax.plot(sub["round"], sub["mean"], label=name)
            ax.fill_between(sub["round"], sub["mean"] - ci, sub["mean"] + ci, alpha=0.2)
        ax.set_title(TITLES[col])
        ax.set_xlabel("Round")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

if __name__ == "__main__":
    df, X = load()
    conditions = {
        "content": lambda s: ContentRecommender(X),
        "random": lambda s: RandomRecommender(len(df), seed=1000 + s),
    }
    print(f"유저 {df['topic'].nunique() * SEEDS}명 x {ROUNDS}라운드 x 조건 {len(conditions)}개")
    result = run_all(df, X, conditions)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_DIR / "day8_baseline.csv", index=False)
    print("\n[후반 5라운드 평균]")
    print(summarize(result).to_string())
    plot(result, Path("docs/figures/day8_baseline.png"), "Content-based vs Random baseline")
    print("\n그래프 저장: docs/figures/day8_baseline.png")
