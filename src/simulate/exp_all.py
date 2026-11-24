from src.recommend.content_based import ContentRecommender, load
from src.rerank.bridging import BridgingReranker, bridging_scores, group_approval
from src.rerank.calibration import CalibratedReranker
from src.rerank.mmr import MMRReranker
from src.simulate.experiment import BIAS, RandomRecommender, run_and_report

if __name__ == "__main__":
    df, X = load()
    base = ContentRecommender(X)
    topics = df["topic"].values
    bridge = bridging_scores(group_approval(X, topics, bias=BIAS))

    print("[브리징 점수 상위 10개 기사]")
    top = bridge.argsort()[::-1][:10]
    print(df.loc[top, ["title", "topic"]].to_string(index=False), "\n")

    conditions = {
        "content": lambda s: base,
        "random": lambda s: RandomRecommender(len(df), seed=1000 + s),
        "mmr_0.5": lambda s: MMRReranker(base, X, lam=0.5),
        "calib_0.6": lambda s:
echo "- 브리징 재랭킹 구현 (집단별 최소 호감도 기반), 전체 기법 통합 비교" > docs/devlog/$(date +%F)-day12.md
git add . && git commit -m "feat: 브리징 재랭킹 구현 및 전체 기법 비교" && git push
mkdir -p src/analysis && touch src/analysis/__init__.py

cat > src/analysis/stats.py << 'EOF'
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

RES_DIR = Path("docs/results")
BASE = "content"
LAST = 5
METRICS = ["topic_entropy", "pref_share", "ild", "utility"]

def holm(pvals) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adj[idx] = running
    return adj

def paired_tests(name: str) -> pd.DataFrame:
    df = pd.read_csv(RES_DIR / f"{name}.csv")
    last_round = df["round"].max()
    tail = df[df["round"] > last_round - LAST].groupby(["condition", "user"])[METRICS].mean()

    rows = []
    for cond in tail.index.get_level_values("condition").unique():
        if cond == BASE:
            continue
        for m in METRICS:
            a = tail.loc[BASE][m]
            b = tail.loc[cond][m].reindex(a.index)
            diff = b - a
            try:
                p = wilcoxon(b, a).pvalue
            except ValueError:
                p = 1.0
            rows.append({
                "condition": cond,
                "metric": m,
                "content_mean": a.mean(),
                "method_mean": b.mean(),
                "change_pct": 100 * diff.mean() / abs(a.mean()),
                "win_rate": (diff > 0).mean(),
                "p": p,
            })
    out = pd.DataFrame(rows)
    out["p_holm"] = holm(out["p"])
    out["sig"] = np.where(out["p_holm"] < 0.001, "***",
                 np.where(out["p_holm"] < 0.01, "**",
                 np.where(out["p_holm"] < 0.05, "*", "")))
    return out.round(4)

if __name__ == "__main__":
    name = sys.argv[1] if len(sys.argv) > 1 else "day12_all"
    out = paired_tests(name)
    out.to_csv(RES_DIR / f"{name}_stats.csv", index=False)

    table = out.assign(cell=out["change_pct"].round(1).astype(str) + "%" + out["sig"])
    print(f"[content 대비 변화율, 유저 짝지은 Wilcoxon + Holm 보정] ({name})")
    print("*** p<0.001  ** p<0.01  * p<0.05\n")
    print(table.pivot(index="condition", columns="metric", values="cell").to_string())
    print(f"\n상세 결과: {RES_DIR}/{name}_stats.csv")
