import os
import json
import re
import html
import hashlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from email.utils import parsedate_to_datetime
from xml.sax.saxutils import escape as xml_escape

import feedparser
import yaml
from google import genai
from google.genai import types


JST = timezone(timedelta(hours=9))

MODEL = "gemini-3.5-flash"

FEEDS_FILE = Path("feeds.yaml")
SEEN_FILE = Path("data/seen.json")
DOCS_DIR = Path("docs")
HISTORY_DIR = DOCS_DIR / "history"

FIRST_RUN_LOOKBACK_HOURS = 12
SEEN_KEEP_DAYS = 14
MAX_ARTICLES_PER_RUN = 120

CATEGORY_ORDER = [
    "主要",
    "国内",
    "国際",
    "経済",
    "IT",
    "科学",
    "地域",
    "BBC Top",
    "BBC World",
    "BBC Business",
]


def clean_text(value):
    if not value:
        return ""

    value = re.sub(
        r"<[^>]+>",
        " ",
        str(value),
    )

    value = html.unescape(value)

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()


def parse_entry_datetime(entry):

    for key in (
        "published_parsed",
        "updated_parsed",
    ):

        parsed = entry.get(key)

        if parsed:
            try:
                return datetime(
                    *parsed[:6],
                    tzinfo=timezone.utc,
                )
            except Exception:
                pass

    for key in (
        "published",
        "updated",
    ):

        raw = entry.get(key)

        if raw:
            try:

                dt = parsedate_to_datetime(
                    raw
                )

                if dt.tzinfo is None:
                    dt = dt.replace(
                        tzinfo=timezone.utc
                    )

                return dt.astimezone(
                    timezone.utc
                )

            except Exception:
                pass

    return datetime.now(
        timezone.utc
    )


def article_id(
    url,
    title,
):

    raw = (
        f"{url}|{title}"
        .encode("utf-8")
    )

    return hashlib.sha256(
        raw
    ).hexdigest()[:16]


def load_feeds():

    with FEEDS_FILE.open(
        "r",
        encoding="utf-8",
    ) as f:

        data = yaml.safe_load(f)

    return data.get(
        "feeds",
        [],
    )


def load_seen():

    if not SEEN_FILE.exists():
        return {}

    try:

        with SEEN_FILE.open(
            "r",
            encoding="utf-8",
        ) as f:

            data = json.load(f)

        if isinstance(
            data,
            dict,
        ):
            return data

    except Exception:
        pass

    return {}


def save_seen(seen):

    SEEN_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(
            days=SEEN_KEEP_DAYS
        )
    )

    cleaned = {}

    for key, value in seen.items():

        try:

            dt = datetime.fromisoformat(
                value
            )

            if dt.tzinfo is None:

                dt = dt.replace(
                    tzinfo=timezone.utc
                )

            if dt >= cutoff:

                cleaned[key] = value

        except Exception:
            continue

    with SEEN_FILE.open(
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            cleaned,
            f,
            ensure_ascii=False,
            indent=2,
        )


def fetch_articles(
    feeds,
    seen,
):

    now = datetime.now(
        timezone.utc
    )

    first_run = not bool(
        seen
    )

    cutoff = (
        now
        - timedelta(
            hours=FIRST_RUN_LOOKBACK_HOURS
        )
    )

    collected = []

    for feed_cfg in feeds:

        parsed = feedparser.parse(
            feed_cfg["url"]
        )

        for entry in parsed.entries:

            title = clean_text(
                entry.get(
                    "title",
                    "",
                )
            )

            url = entry.get(
                "link",
                "",
            ).strip()

            summary = clean_text(
                entry.get("summary")
                or entry.get("description")
                or ""
            )

            if not title or not url:
                continue

            published = (
                parse_entry_datetime(
                    entry
                )
            )

            aid = article_id(
                url,
                title,
            )

            if aid in seen:
                continue

            if (
                first_run
                and published < cutoff
            ):
                continue

            collected.append(
                {
                    "id": aid,
                    "source": feed_cfg[
                        "name"
                    ],
                    "category": feed_cfg[
                        "category"
                    ],
                    "title": title,
                    "summary": summary,
                    "url": url,
                    "published":
                        published.isoformat(),
                }
            )

    deduped = {}

    for article in collected:

        key = article["url"]

        if key not in deduped:

            deduped[key] = article

        else:

            current = deduped[key]

            if (
                current["category"]
                == "主要"
                and article["category"]
                != "主要"
            ):

                deduped[key] = article

    articles = list(
        deduped.values()
    )

    articles.sort(
        key=lambda x:
            x["published"],
        reverse=True,
    )

    return articles[
        :MAX_ARTICLES_PER_RUN
    ]


def build_prompt(
    articles,
):

    compact = [
        {
            "id": a["id"],
            "source": a["source"],
            "category": a["category"],
            "title": a["title"],
            "rss_summary":
                a["summary"],
        }
        for a in articles
    ]

    return f"""
あなたは日本語のニュース編集者です。

以下はRSSから取得した新着記事です。

重要なルール:

- RSSに書かれていない事実を推測・補完しない。
- rss_summary が空の場合、タイトル以上の詳細を勝手に作らない。
- 同じ出来事の重複記事はまとめる。
- 出力は日本語。
- 必ずJSONのみを返す。

ニュースの選定方針:

- 政治、行政、外交、安全保障、経済政策を優先する。
- 政府・国会・中央省庁の重要な動きを優先する。
- 日本経済や企業活動に影響するニュースを優先する。
- 芸能、スポーツ、生活情報、軽い話題は選ばない。
- 事件・事故は全国的な影響や社会的重要性が高いものを優先する。
- 国際ニュースは、日本への影響が大きいもの、国際政治・安全保障・世界経済に重要なものを優先する。
- IT・科学は、政策・産業・社会への影響が大きいニュースを優先する。
- 単なる話題性より、政策・経済・社会への実質的影響を重視する。

水産・海洋ニュースの選定方針:

- 水産業、漁業、養殖業、水産資源、資源管理、海洋政策、水産物貿易に関する重要ニュースを優先する。
- カツオ・マグロ類、国際的な漁業管理、漁獲規制、資源評価、IUU漁業などのニュースは特に重視する。
- 水産関連ニュースは、一般ニュースとしての重要度だけでなく、水産行政・資源管理上の重要性も考慮する。
- 水産・海洋分野で専門的に重要なニュースは、一般社会での話題性が低くても重要ニュースの候補に含める。

「3分で把握」は、
その時間帯に知っておく価値の高いニュースを
最大6項目にまとめてください。

できるだけ以下の分野をバランスよく含めてください。

- 国内政治・行政
- 外交・安全保障
- 経済・経済政策
- 国際
- IT・科学
- 水産・海洋
- その他の重要ニュース

重要なニュースがない分野を
無理に入れる必要はありません。

BBCの記事について:

- BBCの記事は英語見出しを自然な日本語に翻訳する。
- 直訳ではなく、日本語のニュース見出しとして自然で分かりやすくする。
- 固有名詞、数字、事実関係を変更しない。
- 英語原文の見出しも保持する。
- BBC以外の記事は翻訳不要。

形式:

{{
    "quick_summary": [
        "重要ポイント1",
        "重要ポイント2",
        "重要ポイント3",
        "重要ポイント4",
        "重要ポイント5",
        "重要ポイント6"
    ],

    "important": [
        {{
            "id": "元記事のid",
            "headline":
                "分かりやすい日本語見出し",
            "summary":
                "何が起きたか、なぜ重要か、今後の注目点が分かるように2〜4文で簡潔に要約"
        }}
    ],

    "bbc_translations": [
        {{
            "id": "BBC記事のid",
            "ja_title":
                "自然な日本語訳"
        }}
    ]
}}

important は最大10件です。
重要度が低いニュースで無理に10件を埋める必要はありません。

important の id は、
必ず入力記事に存在する id を
そのまま使用してください。

bbc_translations は、
BBC Top、BBC World、BBC Business の記事だけを対象にしてください。

bbc_translations の id も、
必ず入力記事に存在する id を
そのまま使用してください。

同一ニュースを
複数選ばないでください。

入力記事:

{json.dumps(
    compact,
    ensure_ascii=False
)}
""".strip()


def call_gemini(
    articles,
):

    client = genai.Client(
        api_key=os.environ[
            "GEMINI_API_KEY"
        ]
    )

    response = (
        client.models
        .generate_content(
            model=MODEL,
            contents=build_prompt(
                articles
            ),
            config=
                types.GenerateContentConfig(
                    response_mime_type=
                        "application/json",
                    temperature=0.2,
                    max_output_tokens=6500,
                    thinking_config=
                        types.ThinkingConfig(
                            thinking_level=
                                "low"
                        ),
                ),
        )
    )

    data = json.loads(
        response.text
    )

    quick_summary = data.get(
        "quick_summary",
        [],
    )

    important = data.get(
        "important",
        [],
    )

    bbc_translations = data.get(
        "bbc_translations",
        [],
    )

    if not isinstance(
        quick_summary,
        list,
    ):
        quick_summary = []

    if not isinstance(
        important,
        list,
    ):
        important = []

    if not isinstance(
        bbc_translations,
        list,
    ):
        bbc_translations = []

    return {
        "quick_summary":
            quick_summary[:6],
        "important":
            important[:10],
        "bbc_translations":
            bbc_translations,
    }


def category_sort_key(
    category,
):

    try:

        return CATEGORY_ORDER.index(
            category
        )

    except ValueError:

        return len(
            CATEGORY_ORDER
        )


def format_time(
    iso_string,
):

    try:

        dt = datetime.fromisoformat(
            iso_string
        )

        return (
            dt.astimezone(JST)
            .strftime("%H:%M")
        )

    except Exception:

        return ""


def render_quick_summary(
    items,
):

    if not items:
        return ""

    lis = "".join(
        f"<li>{html.escape(str(item))}</li>"
        for item in items
        if str(item).strip()
    )

    if not lis:
        return ""

    return f"""
<section class="quick">

<div class="section-kicker">
QUICK READ
</div>

<h2 class="quick-title">
3分で把握
</h2>

<ul>
{lis}
</ul>

</section>
"""


def render_important(
    important,
    article_map,
):

    cards = []

    for item in important:

        aid = str(
            item.get(
                "id",
                "",
            )
        )

        article = article_map.get(
            aid
        )

        if not article:
            continue

        headline = (
            clean_text(
                item.get(
                    "headline"
                )
            )
            or article["title"]
        )

        summary = (
            clean_text(
                item.get(
                    "summary"
                )
            )
            or article["summary"]
        )

        source = html.escape(
            article["source"]
        )

        category = html.escape(
            article["category"]
        )

        url = html.escape(
            article["url"],
            quote=True,
        )

        time_text = format_time(
            article["published"]
        )

        cards.append(
            f"""
<article class="lead-card">

<div class="eyebrow">
{category}
</div>

<h3>
{html.escape(headline)}
</h3>

<p>
{html.escape(summary)}
</p>

<div class="card-meta">

<span>
{source}
</span>

<span>
{time_text}
</span>

<a
href="{url}"
target="_blank"
rel="noopener"
>
記事を読む ↗
</a>

</div>

</article>
"""
        )

    if not cards:
        return ""

    return f"""
<section class="section">

<div class="section-heading">

<div>

<div class="section-kicker">
TOP STORIES
</div>

<h2>
重要ニュース
</h2>

</div>

</div>

<div class="lead-grid">

{''.join(cards)}

</div>

</section>
"""


def render_other_headlines(
    articles,
    important_ids,
    bbc_translation_map,
):

    remaining = [
        a
        for a in articles
        if a["id"]
        not in important_ids
    ]

    groups = {}

    for article in remaining:

        groups.setdefault(
            article["category"],
            [],
        ).append(article)

    sections = []

    for category in sorted(
        groups.keys(),
        key=category_sort_key,
    ):

        group = groups[
            category
        ]

        rows = []

        for article in group:

            url = html.escape(
                article["url"],
                quote=True,
            )

            source = html.escape(
                article["source"]
            )

            time_text = format_time(
                article["published"]
            )

            is_bbc = (
                article["category"]
                in {
                    "BBC Top",
                    "BBC World",
                    "BBC Business",
                }
            )

            if is_bbc:

                ja_title = (
                    bbc_translation_map.get(
                        article["id"]
                    )
                )

                if ja_title:

                    title_html = f"""
<div class="bbc-ja">
{html.escape(ja_title)}
</div>

<div class="bbc-en">
{html.escape(article["title"])}
</div>
"""

                else:

                    title_html = f"""
<div>
{html.escape(article["title"])}
</div>
"""

            else:

                title_html = f"""
<div>
{html.escape(article["title"])}
</div>
"""

            rows.append(
                f"""
<li>

<a
href="{url}"
target="_blank"
rel="noopener"
>
{title_html}
</a>

<div class="headline-meta">

{source}
・
{time_text}

</div>

</li>
"""
            )

        sections.append(
            f"""
<div class="headline-group">

<h3>
{html.escape(category)}
</h3>

<ul>
{''.join(rows)}
</ul>

</div>
"""
        )

    if not sections:
        return ""

    return f"""
<section class="section other-section">

<div class="section-heading">

<div>

<div class="section-kicker">
LATEST
</div>

<h2>
その他の新着見出し
</h2>

</div>

</div>

<div class="headline-groups">

{''.join(sections)}

</div>

</section>
"""


def page_template(
    body,
    generated_at,
    count,
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

<link rel="apple-touch-icon" sizes="180x180" href="/news-digest/icon.PNG?v=2">

<meta
name="viewport"
content="width=device-width, initial-scale=1"
>

<meta
name="theme-color"
content="#f7f7f5"
>

<title>
News Digest
</title>

<style>

:root {{
    --bg: #f7f7f5;
    --card: #ffffff;
    --text: #161616;
    --sub: #707070;
    --line: #deded9;
    --accent: #b3261e;
}}

@media
(prefers-color-scheme: dark) {{

    :root {{
        --bg: #111211;
        --card: #1a1b1a;
        --text: #f3f3f0;
        --sub: #a6a6a1;
        --line: #333431;
        --accent: #ff8a80;
    }}

}}

* {{
    box-sizing: border-box;
}}

body {{

    margin: 0;

    color: var(--text);

    background:
        var(--bg);

    font-family:
        -apple-system,
        BlinkMacSystemFont,
        "Hiragino Sans",
        "Yu Gothic",
        sans-serif;

    line-height: 1.7;

    -webkit-font-smoothing:
        antialiased;
}}

a {{
    color: inherit;
}}

.topbar {{

    position: sticky;

    top: 0;

    z-index: 20;

    backdrop-filter:
        blur(16px);

    -webkit-backdrop-filter:
        blur(16px);

    background:
        color-mix(
            in srgb,
            var(--bg) 88%,
            transparent
        );

    border-bottom:
        1px solid
        var(--line);
}}

.topbar-inner {{

    max-width: 900px;

    margin: 0 auto;

    min-height: 54px;

    padding:
        0 18px;

    display: flex;

    align-items: center;

    justify-content:
        space-between;
}}

.brand {{

    font-size:
        .96rem;

    font-weight:
        800;

    letter-spacing:
        -.025em;
}}

.update-badge {{

    color:
        var(--sub);

    font-size:
        .74rem;

    font-weight:
        650;
}}

main {{

    max-width:
        900px;

    margin:
        0 auto;

    padding:
        42px 18px 80px;
}}

.hero {{

    padding:
        8px 0 34px;

    border-bottom:
        1px solid
        var(--line);
}}

.hero-label {{

    color:
        var(--accent);

    font-weight:
        800;

    font-size:
        .72rem;

    letter-spacing:
        .12em;
}}

.hero h1 {{

    margin:
        8px 0;

    font-size:
        clamp(
            2.15rem,
            7vw,
            4rem
        );

    line-height:
        1.05;

    letter-spacing:
        -.065em;
}}

.hero-meta {{

    color:
        var(--sub);

    font-size:
        .86rem;
}}

.quick {{

    margin:
        28px 0 44px;

    padding:
        24px 26px;

    background:
        var(--card);

    border:
        1px solid
        var(--line);

    border-radius:
        18px;
}}

.section-kicker {{

    color:
        var(--accent);

    font-size:
        .69rem;

    font-weight:
        850;

    letter-spacing:
        .13em;
}}

.quick-title {{

    margin:
        3px 0 12px;

    font-size:
        1.45rem;

    letter-spacing:
        -.035em;
}}

.quick ul {{

    margin:
        0;

    padding-left:
        1.2rem;
}}

.quick li {{

    margin:
        8px 0;
}}

.section {{

    margin-top:
        46px;
}}

.section-heading {{

    margin-bottom:
        15px;

    padding-bottom:
        10px;

    border-bottom:
        2px solid
        var(--text);
}}

.section-heading h2 {{

    margin:
        2px 0 0;

    font-size:
        1.65rem;

    line-height:
        1.2;

    letter-spacing:
        -.04em;
}}

.lead-grid {{

    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0, 1fr)
        );

    gap:
        12px;
}}

.lead-card {{

    background:
        var(--card);

    border:
        1px solid
        var(--line);

    border-radius:
        17px;

    padding:
        22px;

    min-width:
        0;
}}

.lead-card:first-child {{

    grid-column:
        1 / -1;

    padding:
        28px;
}}

.eyebrow {{

    color:
        var(--accent);

    font-size:
        .72rem;

    font-weight:
        800;

    margin-bottom:
        7px;
}}

.lead-card h3 {{

    margin:
        0 0 11px;

    font-size:
        1.18rem;

    line-height:
        1.48;

    letter-spacing:
        -.025em;
}}

.lead-card:first-child h3 {{

    font-size:
        clamp(
            1.45rem,
            4vw,
            2rem
        );
}}

.lead-card p {{

    margin:
        0;

    font-size:
        .95rem;
}}

.card-meta {{

    margin-top:
        16px;

    padding-top:
        12px;

    border-top:
        1px solid
        var(--line);

    color:
        var(--sub);

    font-size:
        .76rem;

    display:
        flex;

    gap:
        10px;

    align-items:
        center;

    flex-wrap:
        wrap;
}}

.card-meta a {{

    margin-left:
        auto;

    font-weight:
        750;

    text-decoration:
        none;

    color:
        var(--text);
}}

.headline-groups {{

    display:
        grid;

    grid-template-columns:
        repeat(
            2,
            minmax(0, 1fr)
        );

    gap:
        13px;
}}

.headline-group {{

    background:
        var(--card);

    border:
        1px solid
        var(--line);

    border-radius:
        16px;

    padding:
        18px 20px 10px;

    min-width:
        0;
}}

.headline-group h3 {{

    margin:
        0 0 7px;

    padding-bottom:
        9px;

    border-bottom:
        1px solid
        var(--line);

    font-size:
        1rem;
}}

.headline-group ul {{

    list-style:
        none;

    margin:
        0;

    padding:
        0;
}}

.headline-group li {{

    padding:
        11px 0;

    border-bottom:
        1px solid
        var(--line);
}}

.headline-group li:last-child {{

    border-bottom:
        0;
}}

.headline-group a {{

    display:
        block;

    font-weight:
        650;

    font-size:
        .92rem;

    line-height:
        1.55;

    text-decoration:
        none;
}}

.bbc-ja {{

    font-weight:
        700;

    color:
        var(--text);
}}

.bbc-en {{

    margin-top:
        3px;

    color:
        var(--sub);

    font-size:
        .78rem;

    font-weight:
        500;

    line-height:
        1.45;
}}

.headline-meta {{

    color:
        var(--sub);

    margin-top:
        4px;

    font-size:
        .72rem;
}}

.empty {{

    margin-top:
        28px;

    background:
        var(--card);

    border:
        1px solid
        var(--line);

    border-radius:
        17px;

    padding:
        32px 24px;

    text-align:
        center;

    color:
        var(--sub);
}}

.footer {{

    margin-top:
        58px;

    padding-top:
        20px;

    border-top:
        1px solid
        var(--line);

    color:
        var(--sub);

    text-align:
        center;

    font-size:
        .74rem;
}}

@media
(max-width: 650px) {{

    main {{

        padding:
            28px 13px 60px;
    }}

    .topbar-inner {{

        padding:
            0 14px;
    }}

    .quick {{

        padding:
            19px 18px;

        border-radius:
            15px;
    }}

    .lead-grid,
    .headline-groups {{

        grid-template-columns:
            1fr;
    }}

    .lead-card,
    .lead-card:first-child {{

        grid-column:
            auto;

        padding:
            18px;

        border-radius:
            15px;
    }}

    .lead-card:first-child h3 {{

        font-size:
            1.35rem;
    }}

    .headline-group {{

        border-radius:
            15px;
    }}

}}

</style>

</head>

<body>

<header class="topbar">

<div class="topbar-inner">

<div class="brand">
NEWS DIGEST
</div>

<div class="update-badge">
8:07 ・ 12:07 ・ 20:07
</div>

</div>

</header>

<main>

<section class="hero">

<div class="hero-label">
PERSONAL NEWS BRIEFING
</div>

<h1>
今日のニュース
</h1>

<div class="hero-meta">

{generated_text}
更新 ・ 新着
{count}件

</div>

</section>

{body}

<div class="footer">

Yahoo!ニュース・BBCのRSSを取得し、
Geminiで要約しています

</div>

</main>

</body>

</html>
"""


def write_rss(
    articles,
    generated_at,
):

    items = []

    for article in articles[:50]:

        title = xml_escape(
            article["title"]
        )

        link = xml_escape(
            article["url"]
        )

        description = xml_escape(
            article["summary"]
            or article["title"]
        )

        items.append(
            f"""
<item>

<title>
{title}
</title>

<link>
{link}
</link>

<guid>
{link}
</guid>

<description>
{description}
</description>

</item>
"""
        )

    rss = f"""<?xml version="1.0" encoding="UTF-8"?>

<rss version="2.0">

<channel>

<title>
News Digest
</title>

<description>
Personal News Digest
</description>

<link>
./
</link>

<lastBuildDate>
{generated_at.astimezone(timezone.utc).strftime("%a, %d %b %Y %H:%M:%S +0000")}
</lastBuildDate>

{''.join(items)}

</channel>

</rss>
"""

    (
        DOCS_DIR
        / "feed.xml"
    ).write_text(
        rss,
        encoding="utf-8",
    )


def main():

    generated_at = datetime.now(
        JST
    )

    DOCS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    HISTORY_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feeds = load_feeds()

    seen = load_seen()

    articles = fetch_articles(
        feeds,
        seen,
    )

    if not articles:

        body = """
<div class="empty">

今回の更新では
新着記事はありませんでした。

</div>
"""

        page = page_template(
            body,
            generated_at,
            0,
        )

        (
            DOCS_DIR
            / "index.html"
        ).write_text(
            page,
            encoding="utf-8",
        )

        return

    digest = call_gemini(
        articles
    )

    article_map = {
        a["id"]: a
        for a in articles
    }

    important_ids = {

        str(
            item.get(
                "id",
                "",
            )
        )

        for item
        in digest["important"]

        if str(
            item.get(
                "id",
                "",
            )
        )
        in article_map
    }

    bbc_translation_map = {}

    for item in digest[
        "bbc_translations"
    ]:

        aid = str(
            item.get(
                "id",
                "",
            )
        )

        ja_title = clean_text(
            item.get(
                "ja_title",
                "",
            )
        )

        if aid and ja_title:

            bbc_translation_map[
                aid
            ] = ja_title

    body = (

        render_quick_summary(
            digest[
                "quick_summary"
            ]
        )

        + render_important(
            digest[
                "important"
            ],
            article_map,
        )

        + render_other_headlines(
            articles,
            important_ids,
            bbc_translation_map,
        )
    )

    page = page_template(
        body,
        generated_at,
        len(articles),
    )

    (
        DOCS_DIR
        / "index.html"
    ).write_text(
        page,
        encoding="utf-8",
    )

    history_name = (
        generated_at.strftime(
            "%Y-%m-%d-%H%M.html"
        )
    )

    (
        HISTORY_DIR
        / history_name
    ).write_text(
        page,
        encoding="utf-8",
    )

    write_rss(
        articles,
        generated_at,
    )

    now_iso = (
        datetime.now(
            timezone.utc
        )
        .isoformat()
    )

    for article in articles:

        seen[
            article["id"]
        ] = now_iso

    save_seen(
        seen
    )


if __name__ == "__main__":
    main()
