import itertools

import pandas as pd

import src.simulate.experiment as exp
from src.recommend.content_based import ContentRecommender, load
from src.rerank.bridging import BridgingReranker, bridging_scores, group_approval
from src.rerank.calibration import CalibratedReranker
from src.rerank.mmr import MMRReranker

BIASES = [0.1, 0.3, 0.5]
TEMPS = [0.05, 0.1, 0.2]
COLS = ["topic_entropy", "pref_share", "utility"]

if __name__ == "__main__":
    df, X = load()
    base = ContentRecommender(X)
    topics = df["topic"].values
    exp.SEEDS = 5

    rows = []
    settings = list(itertools.product(BIASES, TEMPS))
    for i, (bias, temp) in enumerate(settings, 1):
        print(f"\n=== [{i}/{len(settings)}] BIAS={bias}, TEMPERATURE={temp} ===")
        exp.BIAS, exp.TEMPERATURE = bias, temp
        bridge = bridging_scores(group_approval(X, topics, bias=bias))
        conditions = {
            "content": lambda s: base,
            "mmr_0.5": lambda s: MMRReranker(base, X, lam=0.5),
            "calib_0.6": lambda s: CalibratedReranker(base, topics, lam=0.6),
            "bridge_0.5": lambda s, b=bridge: BridgingReranker(base, b, lam=0.5),
        }
        res = exp.run_all(df, X, conditions)
        tail = res[res["round"] > exp.ROUNDS - 5].groupby("condition")[COLS].mean()
        for cond, row in tail.iterrows():
            rows.append({"bias": bias, "temp": temp, "condition": cond, **row.to_dict()})

    out = pd.DataFrame(rows)
    ref = out[out["condition"] == "content"][["bias", "temp", "topic_entropy", "utility"]]
    out = out.merge(ref.rename(columns={"topic_entropy": "e0", "utility": "u0"}), on=["bias", "temp"])
    out["entropy_gain_pct"] = 100 * (out["topic_entropy"] - out["e0"]) / out["e0"]
    out["utility_loss_pct"] = 100 * (out["u0"] - out["utility"]) / out["u0"]
    out = out.drop(columns=["e0", "u0"]).round(3)
    exp.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(exp.OUT_DIR / "day14_sensitivity.csv", index=False)

    methods = out[out["condition"] != "content"]
    print("\n[content 대비 주제 다양성 증가율 %]")
    print(methods.pivot_table(index=["bias", "temp"], columns="condition", values="entropy_gain_pct").round(1).to_string())
    print("\n[content 대비 만족도 손실률 %]")
    print(methods.pivot_table(index=["bias", "temp"], columns="condition", values="utility_loss_pct").round(1).to_string())
