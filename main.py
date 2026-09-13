import os
import re
import json
import html
import hashlib
from pathlib import Path
from datetime import datetime, timezone, timedelta

import feedparser
import yaml
from google import genai
from google.genai import types


# ===== 基本設定 =====

JST = timezone(timedelta(hours=9))

# Gemini無料Tierを使用
MODEL = "gemini-3.8-flash"

# 1回にAIへ送る記事数の絶対上限
MAX_ARTICLES_PER_RUN = 120

# RSS本文・概要が長い場合の上限
MAX_SUMMARY_CHARS = 900

# 初回実行時は過去12時間分まで
FIRST_RUN_LOOKBACK_HOURS = 12

# 処理済み記事の記録を14日間保持
SEEN_KEEP_DAYS = 14


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DOCS_DIR = ROOT / "docs"
HISTORY_DIR = DOCS_DIR / "history"

SEEN_FILE = DATA_DIR / "seen.json"
FEEDS_FILE = ROOT / "feeds.yaml"


# ===== 文字列処理 =====

def clean_text(value: str) -> str:
    if not value:
        return ""

    # RSS概要に含まれるHTMLタグを除去
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def parse_entry_time(entry):
    for key in ("published_parsed", "updated_parsed", "created_parsed"):
        value = entry.get(key)
        if value:
            return datetime(*value[:6], tzinfo=timezone.utc)

    return None


def article_key(link: str, title: str) -> str:
    base = (link or "").strip() or title.strip()
    return hashlib.sha256(
        base.encode("utf-8")
    ).hexdigest()[:20]


# ===== 設定・処理済み記事を読み込み =====

def load_feeds():
    with FEEDS_FILE.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    return data.get("feeds", [])


def load_seen():
    if not SEEN_FILE.exists():
        return {}

    try:
        with SEEN_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)

        return data if isinstance(data, dict) else {}

    except (json.JSONDecodeError, OSError):
        return {}


def save_seen(seen):
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cutoff = datetime.now(timezone.utc) - timedelta(
        days=SEEN_KEEP_DAYS
    )

    pruned = {}

    for key, iso in seen.items():
        try:
            dt = datetime.fromisoformat(iso)

            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)

            if dt >= cutoff:
                pruned[key] = iso

        except Exception:
            continue

    with SEEN_FILE.open("w", encoding="utf-8") as f:
        json.dump(
            pruned,
            f,
            ensure_ascii=False,
            indent=2
        )


# ===== RSS取得 =====

def fetch_articles(feeds, seen):
    now = datetime.now(timezone.utc)

    first_run_cutoff = now - timedelta(
        hours=FIRST_RUN_LOOKBACK_HOURS
    )

    is_first_run = len(seen) == 0

    articles = []
    current_keys = set()

    for feed in feeds:
        name = feed["name"]
        category = feed["category"]
        url = feed["url"]

        print(f"[INFO] Reading: {name}")

        parsed = feedparser.parse(
            url,
            request_headers={
                "User-Agent":
                "news-digest/1.0 (+GitHub Actions; RSS reader)"
            }
        )

        if getattr(parsed, "bozo", False):
            print(
                f"[WARN] Feed parse warning: "
                f"{name}: {parsed.bozo_exception}"
            )

        for entry in parsed.entries:

            title = clean_text(
                entry.get("title", "")
            )

            link = (
                entry.get("link", "") or ""
            ).strip()

            summary = clean_text(
                entry.get("summary", "")
                or entry.get("description", "")
            )[:MAX_SUMMARY_CHARS]

            if not title:
                continue

            key = article_key(link, title)

            # 今回の取得内で重複
            if key in current_keys:
                continue

            # 過去に処理済み
            if key in seen:
                continue

            published = parse_entry_time(entry)

            # 初回だけ過去12時間より古いものを除外
            if (
                is_first_run
                and published
                and published < first_run_cutoff
            ):
                continue

            current_keys.add(key)

            articles.append({
                "key": key,
                "source": name,
                "category": category,
                "title": title,
                "summary": summary,
                "url": link,
                "published":
                    published.isoformat()
                    if published
                    else "",
            })

    # 新しい順
    articles.sort(
        key=lambda a: a["published"] or "",
        reverse=True
    )

    # 無料枠保護のため絶対上限
    return articles[:MAX_ARTICLES_PER_RUN]


# ===== Geminiへ送るデータ =====

def prompt_payload(articles):

    payload = []

    for idx, article in enumerate(
        articles,
        start=1
    ):
        article["id"] = f"A{idx:03d}"

        payload.append({
            "id": article["id"],
            "source": article["source"],
            "category": article["category"],
            "title": article["title"],
            "rss_summary": article["summary"],
            "published": article["published"],
        })

    return payload


# ===== Geminiで要約 =====

def summarize_with_gemini(articles):

    api_key = os.environ.get(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set."
        )

    payload = prompt_payload(articles)

    prompt = f"""
あなたは日本語のニュース編集者です。

以下はYahoo!ニュースとBBCのRSSから取得した
新着ニュースです。

【重要なルール】

・RSSに書かれている情報だけを使用してください。
・RSSにない事実を推測・補完しないでください。
・rss_summary が空の場合は、タイトル以上の内容を
  断定しないでください。
・同じ出来事を扱う複数記事は可能な範囲で統合してください。
・BBCの英語見出し・概要は自然な日本語にしてください。
・重要ニュースは最大10件にしてください。
・それ以外はカテゴリ別に整理してください。
・要約するニュースは全体で最大25件にしてください。
・要約しなかった記事は other_article_ids に入れてください。
・article_idsには根拠として使った記事IDだけを入れてください。
・URLは出力しないでください。
・各ニュースの要約は原則2〜4文程度にしてください。
・日本の読者がニュース全体を短時間で把握できるようにしてください。

次のJSON形式だけを返してください。

{{
  "overview":
    "この時間帯のニュース全体を2〜4文でまとめる",

  "sections": [

    {{
      "name": "重要ニュース",
      "items": [
        {{
          "title": "日本語の見出し",
          "summary": "ニュースの要約",
          "article_ids": [
            "A001",
            "A002"
          ]
        }}
      ]
    }},

    {{
      "name": "国内",
      "items": []
    }},

    {{
      "name": "国際",
      "items": []
    }},

    {{
      "name": "経済",
      "items": []
    }},

    {{
      "name": "IT・科学",
      "items": []
    }},

    {{
      "name": "スポーツ",
      "items": []
    }},

    {{
      "name": "エンタメ・ライフ・地域",
      "items": []
    }}
  ],

  "other_article_ids": [
    "A026"
  ]
}}

RSSデータ:

{json.dumps(payload, ensure_ascii=False)}
"""

    client = genai.Client(
        api_key=api_key
    )

    # Gemini APIは1回の実行につき1回だけ呼び出す
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(

            response_mime_type=
                "application/json",

            temperature=0.2,

            max_output_tokens=7000,

            # 無料枠節約・高速化
            thinking_config=
                types.ThinkingConfig(
                    thinking_level="low"
                ),
        ),
    )

    if not response.text:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return json.loads(response.text)


# ===== 記事への元リンクを生成 =====

def source_links(
    article_ids,
    article_map
):

    links = []
    seen_urls = set()

    for article_id in article_ids:

        article = article_map.get(
            article_id
        )

        if not article:
            continue

        url = article.get("url", "")
        label = article.get(
            "source",
            "元記事"
        )

        if (
            url
            and url not in seen_urls
        ):

            links.append(
                f'<a href="'
                f'{html.escape(url, quote=True)}'
                f'" target="_blank" '
                f'rel="noopener">'
                f'{html.escape(label)}'
                f'</a>'
            )

            seen_urls.add(url)

    return " / ".join(links)


# ===== ニュース本文をHTML化 =====

def render_digest_html(
    result,
    articles
):

    article_map = {
        a["id"]: a
        for a in articles
    }

    parts = [
        "<p class='overview'>"
        + html.escape(
            result.get(
                "overview",
                ""
            )
        )
        + "</p>"
    ]

    for section in result.get(
        "sections",
        []
    ):

        items = section.get(
            "items",
            []
        )

        if not items:
            continue

        parts.append(
            "<h2>"
            + html.escape(
                section.get(
                    "name",
                    "ニュース"
                )
            )
            + "</h2>"
        )

        for item in items:

            title = html.escape(
                item.get(
                    "title",
                    ""
                )
            )

            summary = html.escape(
                item.get(
                    "summary",
                    ""
                )
            )

            ids = item.get(
                "article_ids",
                []
            )

            sources = source_links(
                ids,
                article_map
            )

            parts.append(
                "<article class='news-item'>"
            )

            parts.append(
                f"<h3>{title}</h3>"
            )

            parts.append(
                f"<p>{summary}</p>"
            )

            if sources:
                parts.append(
                    "<p class='sources'>"
                    f"出典: {sources}"
                    "</p>"
                )

            parts.append(
                "</article>"
            )

    # AI要約対象外の記事は見出しだけ表示
    other_ids = result.get(
        "other_article_ids",
        []
    )

    if other_ids:

        parts.append(
            "<h2>その他の新着見出し</h2>"
            "<ul class='other-list'>"
        )

        for article_id in other_ids:

            article = article_map.get(
                article_id
            )

            if not article:
                continue

            title = html.escape(
                article["title"]
            )

            url = html.escape(
                article.get(
                    "url",
                    ""
                ),
                quote=True
            )

            source = html.escape(
                article["source"]
            )

            if url:

                parts.append(
                    f"<li>"
                    f"<a href='{url}' "
                    f"target='_blank' "
                    f"rel='noopener'>"
                    f"{title}</a>"
                    f" <span>— {source}</span>"
                    f"</li>"
                )

            else:

                parts.append(
                    f"<li>"
                    f"{title}"
                    f" <span>— {source}</span>"
                    f"</li>"
                )

        parts.append("</ul>")

    return "\n".join(parts)


# ===== Webページ全体 =====

def page_template(
    body,
    generated_at,
    count
):

    generated_text = (
        generated_at
        .astimezone(JST)
        .strftime(
            "%Y年%m月%d日 %H:%M"
        )
    )

    return f"""<!doctype html>
<html lang="ja">

<head>

<meta charset="utf-8">

<meta name="viewport"
content="width=device-width, initial-scale=1">

<title>News Digest</title>

<style>

:root {{
  color-scheme: light dark;
  --bg: #f6f7f8;
  --card: #ffffff;
  --text: #171717;
  --sub: #666;
  --line: #e5e5e5;
}}

@media (prefers-color-scheme: dark) {{

  :root {{
    --bg: #111214;
    --card: #1b1d20;
    --text: #f2f2f2;
    --sub: #aaa;
    --line: #303236;
  }}

}}

body {{
  margin: 0;
  background: var(--bg);
  color: var(--text);

  font-family:
    -apple-system,
    BlinkMacSystemFont,
    "Hiragino Sans",
    "Yu Gothic",
    sans-serif;

  line-height: 1.75;
}}

main {{
  max-width: 760px;
  margin: 0 auto;
  padding: 24px 16px 60px;
}}

header {{
  margin-bottom: 24px;
}}

h1 {{
  font-size: 1.8rem;
  margin-bottom: 4px;
}}

.meta,
.sources,
.other-list span {{
  color: var(--sub);
  font-size: .9rem;
}}

.overview {{
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
}}

h2 {{
  margin-top: 34px;
  border-bottom:
    1px solid var(--line);
  padding-bottom: 8px;
}}

.news-item {{
  background: var(--card);
  border:
    1px solid var(--line);
  border-radius: 14px;
  padding: 16px;
  margin: 12px 0;
}}

.news-item h3 {{
  margin: 0 0 8px;
  font-size: 1.08rem;
}}

.news-item p {{
  margin: 6px 0;
}}

a {{
  color: inherit;
  text-decoration-thickness: .08em;
  text-underline-offset: .15em;
}}

.other-list li {{
  margin: 10px 0;
}}

</style>

</head>

<body>

<main>

<header>

<h1>News Digest</h1>

<div class="meta">
{generated_text} 更新 ・ 新着 {count}件
</div>

</header>

{body}

</main>

</body>

</html>
"""


# ===== GitHub PagesのURL =====

def make_site_url():

    repo = os.environ.get(
        "GITHUB_REPOSITORY",
        ""
    )

    if "/" not in repo:
        return ""

    owner, name = repo.split(
        "/",
        1
    )

    return (
        f"https://{owner}.github.io/"
        f"{name}/"
    )


# ===== HTMLとRSSを保存 =====

def write_outputs(
    result,
    articles
):

    generated_at = datetime.now(
        timezone.utc
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    body = render_digest_html(
        result,
        articles
    )

    page = page_template(
        body,
        generated_at,
        len(articles)
    )

    # 最新版
    latest_path = (
        DOCS_DIR
        / "index.html"
    )

    latest_path.write_text(
        page,
        encoding="utf-8"
    )

    # 過去分
    history_name = (
        generated_at
        .astimezone(JST)
        .strftime(
            "%Y-%m-%d-%H%M.html"
        )
    )

    history_path = (
        HISTORY_DIR
        / history_name
    )

    history_path.write_text(
        page,
        encoding="utf-8"
    )

    # ===== RSSフィード生成 =====

    site_url = make_site_url()

    if site_url:

        item_url = (
            f"{site_url}"
            f"history/"
            f"{history_name}"
        )

    else:

        item_url = (
            f"history/"
            f"{history_name}"
        )

    pub_date = generated_at.strftime(
        "%a, %d %b %Y "
        "%H:%M:%S +0000"
    )

    title = (
        generated_at
        .astimezone(JST)
        .strftime(
            "%Y/%m/%d %H:%M "
            "ニュースまとめ"
        )
    )

    description = html.escape(
        result.get(
            "overview",
            ""
        )
    )

    feed_path = (
        DOCS_DIR
        / "feed.xml"
    )

    existing_items = []

    if feed_path.exists():

        text = feed_path.read_text(
            encoding="utf-8"
        )

        existing_items = re.findall(
            r"<item>.*?</item>",
            text,
            flags=re.S
        )[:29]

    new_item = f"""
<item>

<title>
{html.escape(title)}
</title>

<link>
{html.escape(item_url)}
</link>

<guid>
{html.escape(item_url)}
</guid>

<pubDate>
{pub_date}
</pubDate>

<description>
{description}
</description>

</item>
"""

    channel_link = (
        site_url
        or "./"
    )

    feed_xml = f"""<?xml version="1.0" encoding="UTF-8"?>

<rss version="2.0">

<channel>

<title>News Digest</title>

<link>
{html.escape(channel_link)}
</link>

<description>
Yahoo!ニュースとBBCの自動ニュース要約
</description>

<language>ja</language>

{new_item}

{''.join(existing_items)}

</channel>

</rss>
"""

    feed_path.write_text(
        feed_xml,
        encoding="utf-8"
    )


# ===== メイン処理 =====

def main():

    feeds = load_feeds()
    seen = load_seen()

    print(
        f"[INFO] Loaded "
        f"{len(feeds)} feeds."
    )

    articles = fetch_articles(
        feeds,
        seen
    )

    print(
        f"[INFO] New articles: "
        f"{len(articles)}"
    )

    if not articles:

        print(
            "[INFO] No new articles. "
            "Nothing to do."
        )

        return

    result = summarize_with_gemini(
        articles
    )

    write_outputs(
        result,
        articles
    )

    now_iso = datetime.now(
        timezone.utc
    ).isoformat()

    for article in articles:
        seen[
            article["key"]
        ] = now_iso

    save_seen(seen)

    print(
        "[INFO] Digest generated "
        "successfully."
    )


if __name__ == "__main__":
    main()
