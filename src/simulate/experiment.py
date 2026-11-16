from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.metrics.diversity import intra_list_diversity, normalized_entropy
from src.metrics.population import population_metrics
from src.recommend.content_based import ContentRecommender, load

ROUNDS = 30
K = 10
N_INIT = 5
SEEDS = 10
BIAS = 0.3
TEMPERATURE = 0.1
OUT_DIR = Path("docs/results")
FIG_DIR = Path("docs/figures")
METRICS = ["topic_entropy", "pref_share", "ild", "utility"]
TITLES = {
    "topic_entropy": "Topic entropy (higher = diverse)",
    "pref_share": "Share of preferred topic",
    "ild": "Intra-list diversity",
    "utility": "User utility of feed",
    "homogenization": "Feed overlap between users",
    "coverage": "Catalog coverage (cumulative)",
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
        self.taste = ContentRecommender(X)
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
        feed = [int(i) for i in rec.recommend(history, k=K)]
        u = user.utility(history, feed)
        p = np.exp((u - u.max()) / TEMPERATURE)
        p /= p.sum()
        log.append({
            "round": r,
            "feed": feed,
            "topic_entropy": normalized_entropy(topics[feed], n_topics),
            "pref_share": float((topics[feed] == pref_topic).mean()),
            "ild": intra_list_diversity(X, feed),
            "utility": float(u.mean()),
        })
        history.append(int(rng.choice(feed, p=p)))
    return pd.DataFrame(log)

def run_all(df, X, conditions: dict) -> pd.DataFrame:
    """conditions: {조건 이름: 유저 시드를 받아 추천기를 돌려주는 함수}
    같은 유저는 모든 조건에서 같은 시드를 써서 짝지어 비교됨
    """
    runs = []
    topics = list(df["topic"].unique())
    for name, make_rec in conditions.items():
        for t_idx, topic in enumerate(topics):
            for s in range(SEEDS):
                user_seed = t_idx * 100 + s
                res = simulate(df, X, make_rec(user_seed), topic, user_seed)
                runs.append(res.assign(condition=name, user=f"{topic}-{s}"))
        print(f"  {name} 완료")
    return pd.concat(runs, ignore_index=True)

def summarize(result: pd.DataFrame, pop: pd.DataFrame, last: int = 5) -> pd.DataFrame:
    tail = result[result["round"] > ROUNDS - last]
    s = tail.groupby("condition")[METRICS].mean()
    s["homogenization"] = pop[pop["round"] > ROUNDS - last].groupby("condition")["homogenization"].mean()
    s["coverage"] = pop[pop["round"] == ROUNDS].set_index("condition")["coverage"]
    return s.round(3)

def plot_curves(data: pd.DataFrame, cols: list[str], out_path: Path, title: str):
    fig, axes = plt.subplots(1, len(cols), figsize=(4.5 * len(cols), 4))
    axes = np.atleast_1d(axes)
    for ax, col in zip(axes, cols):
        g = data.groupby(["condition", "round"])[col].agg(["mean", "std", "count"]).reset_index()
        for name, sub in g.groupby("condition"):
            ax.plot(sub["round"], sub["mean"], label=name)
            if sub["count"].max() > 1:
                ci = 1.96 * sub["std"].fillna(0) / np.sqrt(sub["count"])
                ax.fill_between(sub["round"], sub["mean"] - ci, sub["mean"] + ci, alpha=0.15)
        ax.set_title(TITLES[col])
        ax.set_xlabel("Round")
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_tradeoff(summary: pd.DataFrame, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for ax, col in zip(axes, ["topic_entropy", "homogenization"]):
        ax.scatter(summary["utility"], summary[col])
        for name, row in summary.iterrows():
            ax.annotate(name, (row["utility"], row[col]),
                        textcoords="offset points", xytext=(5, 5), fontsize=8)
        ax.set_xlabel("User utility (higher = more satisfied)")
        ax.set_ylabel(TITLES[col])
    fig.suptitle("Trade-off: utility vs diversity / homogenization")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def run_and_report(df, X, conditions: dict, name: str, title: str) -> pd.DataFrame:
    print(f"[{name}] 유저 {df['topic'].nunique() * SEEDS}명 x {ROUNDS}라운드 x 조건 {len(conditions)}개")
    result = run_all(df, X, conditions)
    pop = population_metrics(result, len(df))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result.drop(columns="feed").to_csv(OUT_DIR / f"{name}.csv", index=False)
    pop.to_csv(OUT_DIR / f"{name}_population.csv", index=False)
    summary = summarize(result, pop)
    summary.to_csv(OUT_DIR / f"{name}_summary.csv")
    print("\n[후반 5라운드 평균 / 커버리지는 최종값]")
    print(summary.to_string())

    plot_curves(result, METRICS, FIG_DIR / f"{name}_curves.png", title)
    plot_curves(pop, ["homogenization", "coverage"], FIG_DIR / f"{name}_population.png", f"{title} (population)")
    plot_tradeoff(summary, FIG_DIR / f"{name}_tradeoff.png")
    print(f"\n그래프 저장: {FIG_DIR}/{name}_*.png")
    return summary

if __name__ == "__main__":
    df, X = load()
    conditions = {
        "content": lambda s: ContentRecommender(X),
        "random": lambda s: RandomRecommender(len(df), seed=1000 + s),
    }
    run_and_report(df, X, conditions, "day10_baseline", "Content-based vs Random")
