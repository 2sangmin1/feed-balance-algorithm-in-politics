import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
OUT_PATH = Path("data/processed/articles.csv")

def load_raw() -> pd.DataFrame:
    rows = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return pd.DataFrame(rows)

def clean_title(title: str, source: str) -> str:
    suffix = f" - {source}"
    if source and title.endswith(suffix):
        title = title[: -len(suffix)]
    return title.strip()

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["source"] = df["source"].fillna("")
    df["title"] = [clean_title(t, s) for t, s in zip(df["title"], df["source"])]
    df = df[df["title"].str.len() >= 5]
    df = df.drop_duplicates(subset="title").reset_index(drop=True)
    df.insert(0, "id", df.index)
    return df

if __name__ == "__main__":
    raw = load_raw()
    df = preprocess(raw)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"원본 {len(raw)}건 → 정제 후 {len(df)}건")
    print("\n[주제별 기사 수]")
    print(df["topic"].value_counts().to_string())
    print(f"\n[언론사 수] {df['source'].nunique()}곳")
    print("[상위 15개 언론사]")
    print(df["source"].value_counts().head(15).to_string())
