import json
from datetime import date
from pathlib import Path
from urllib.parse import quote

import feedparser

TOPICS = ["경제", "부동산", "교육", "외교", "복지", "노동", "환경", "국회"]

def google_news_url(query: str) -> str:
    return f"https://news.google.com/rss/search?q={quote(query)}&hl=ko&gl=KR&ceid=KR:ko"

def collect() -> list[dict]:
    articles, seen = [], set()
    for topic in TOPICS:
        feed = feedparser.parse(google_news_url(topic))
        for entry in feed.entries:
            if entry.link in seen:
                continue
            seen.add(entry.link)
            articles.append({
                "topic": topic,
                "title": entry.title,
                "source": entry.get("source", {}).get("title", ""),
                "link": entry.link,
                "published": entry.get("published", ""),
            })
        print(f"[{topic}] 누적 {len(articles)}건")
    return articles

def save(articles: list[dict]) -> Path:
    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{date.today()}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for a in articles:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    return path

if __name__ == "__main__":
    path = save(collect())
    print(f"저장 완료: {path}")
