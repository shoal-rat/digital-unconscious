"""Turn a raw (app, window title, url) sample into a *subject*: the thing that
was attended to. Subjects are what the rest of the system reasons about.

Two samples with the same subject key merge into one trace, so the key must be
stable across trivial title changes (unread counters, unsaved-file dots, page
numbers) while still separating genuinely different things.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from urllib.parse import parse_qs, urlsplit

from unconscious.sense.privacy import clean_url, domain_of, is_private_app, is_quiet, redact

BROWSERS = (
    "google chrome", "chrome", "safari", "microsoft edge", "edge", "firefox", "arc", "brave",
    "opera", "vivaldi", "chromium", "orion", "zen", "dia", "duckduckgo", "sidekick",
)

APP_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("system", (
        "finder", "system settings", "system preferences", "activity monitor", "loginwindow",
        "screensaver", "dock", "control center", "notification center", "spotlight", "raycast",
        "alfred", "installer", "app store", "explorer.exe", "taskmgr", "settings",
    )),
    ("chat", (
        "slack", "messages", "wechat", "微信", "mail", "telegram", "discord", "zoom", "teams",
        "outlook", "whatsapp", "signal", "lark", "飞书", "qq", "dingtalk", "钉钉", "facetime", "spark",
    )),
    ("code", (
        "visual studio code", "code", "cursor", "xcode", "pycharm", "intellij", "webstorm", "sublime",
        "zed", "terminal", "iterm", "warp", "ghostty", "alacritty", "kitty", "wezterm", "nova",
        "vim", "emacs", "windsurf", "android studio",
    )),
    ("data", (
        "excel", "numbers", "stata", "rstudio", "positron", "spss", "tableau", "jupyter", "matlab",
        "eviews", "datagrip", "tableplus", "sas", "mathematica",
    )),
    ("reading", (
        "preview", "skim", "zotero", "pdf expert", "books", "kindle", "readcube", "papers",
        "highlights", "marginnote", "acrobat", "mendeley", "readwise", "reeder", "netnewswire",
    )),
    ("writing", (
        "word", "pages", "notion", "obsidian", "bear", "notes", "typora", "ulysses", "scrivener",
        "logseq", "craft", "textedit", "ia writer", "drafts", "keynote", "powerpoint", "tana",
    )),
    ("design", ("figma", "sketch", "photoshop", "illustrator", "affinity", "pixelmator", "blender")),
    ("media", ("spotify", "music", "vlc", "iina", "quicktime", "netflix", "podcasts", "网易云")),
    ("ai", ("chatgpt", "claude", "perplexity", "gemini")),
]

DOMAIN_CATEGORIES: list[tuple[str, tuple[str, ...]]] = [
    ("chat", (
        "mail.google.com", "outlook.live.com", "outlook.office.com", "web.whatsapp.com",
        "app.slack.com", "web.telegram.org", "discord.com", "wx.qq.com", "mail.qq.com",
        "teams.microsoft.com", "messenger.com",
    )),
    ("reading", (
        "arxiv.org", "scholar.google.", "ssrn.com", "nber.org", "jstor.org", "sciencedirect.com",
        "springer.com", "wiley.com", "tandfonline.com", "nature.com", "science.org", "pnas.org",
        "aeaweb.org", "researchgate.net", "semanticscholar.org", "openalex.org", "pubmed",
        "ncbi.nlm.nih.gov", "doi.org", "sagepub.com", "academic.oup.com", "cambridge.org",
        "biorxiv.org", "medrxiv.org", "cnki.net", "wikipedia.org", "substack.com", "medium.com",
        "lesswrong.com", "news.ycombinator.com", "informs.org", "hbr.org", "economist.com",
    )),
    ("code", (
        "github.com", "gitlab.com", "stackoverflow.com", "stackexchange.com", "huggingface.co",
        "docs.python.org", "pypi.org", "npmjs.com", "developer.mozilla.org", "readthedocs.io",
    )),
    ("data", ("kaggle.com", "colab.research.google.com", "data.gov", "fred.stlouisfed.org", "ourworldindata.org")),
    ("writing", ("docs.google.com", "notion.so", "overleaf.com", "feishu.cn", "yuque.com", "craft.do")),
    ("media", (
        "youtube.com", "bilibili.com", "netflix.com", "open.spotify.com", "twitch.tv", "douyin.com",
        "tiktok.com", "iqiyi.com", "youku.com", "vimeo.com",
    )),
    ("social", (
        "x.com", "twitter.com", "reddit.com", "weibo.com", "xiaohongshu.com", "linkedin.com",
        "facebook.com", "instagram.com", "threads.net", "zhihu.com", "douban.com", "bsky.app",
    )),
    ("ai", ("chatgpt.com", "chat.openai.com", "claude.ai", "perplexity.ai", "gemini.google.com", "kimi.com", "deepseek.com")),
]

# host fragment -> (path prefix, query parameter)
SEARCH_ENGINES: list[tuple[str, str, tuple[str, ...]]] = [
    ("scholar.google.", "/scholar", ("q",)),
    ("google.", "/search", ("q",)),
    ("bing.com", "/search", ("q",)),
    ("duckduckgo.com", "/", ("q",)),
    ("baidu.com", "/s", ("wd", "word")),
    ("sogou.com", "/", ("query",)),
    ("yandex.", "/search", ("text",)),
    ("search.yahoo.", "/search", ("p",)),
    ("kagi.com", "/search", ("q",)),
    ("perplexity.ai", "/search", ("q",)),
    ("youtube.com", "/results", ("search_query",)),
    ("search.bilibili.com", "/", ("keyword",)),
    ("github.com", "/search", ("q",)),
    ("x.com", "/search", ("q",)),
    ("twitter.com", "/search", ("q",)),
    ("reddit.com", "/search", ("q",)),
    ("zhihu.com", "/search", ("q",)),
    ("arxiv.org", "/search", ("query",)),
    ("semanticscholar.org", "/search", ("q",)),
    ("openalex.org", "/", ("search",)),
    ("wikipedia.org", "/w/index.php", ("search",)),
]

TITLE_SEARCH_PATTERNS = [
    re.compile(r"^(?P<q>.+?) - Google (?:Search|搜索|Scholar|学术搜索)$"),
    re.compile(r"^(?P<q>.+?) - (?:Bing|必应)(?: Search)?$"),
    re.compile(r"^(?P<q>.+?) at DuckDuckGo$"),
    re.compile(r"^(?P<q>.+?)_百度搜索$"),
    re.compile(r"^(?P<q>.+?) - (?:Search|搜索) - (?:Perplexity|Kagi)$"),
]

_BROWSER_SUFFIX = re.compile(
    r"\s+[-–—|]\s+(?:Google Chrome|Microsoft​?\s?Edge|Mozilla Firefox|Firefox|Safari|Brave|Arc|"
    r"Opera|Vivaldi|Chromium|Orion)\s*$",
    re.I,
)
_PROFILE_SUFFIX = re.compile(r"\s+[-–—]\s+(?:Personal|Work|Default|Profile \d+|个人|工作)\s*$", re.I)
_MORE_PAGES = re.compile(r"\s+(?:and \d+ more pages?|和另外 ?\d+ ?个页面)\s*$", re.I)
_COUNTER = re.compile(r"^\s*[\(\[]\d+[\)\]]\s*|^\s*[•●◉*]\s*")
_EDITED = re.compile(r"\s*(?:[-–—]\s*Edited|\(Edited\)|[-–—]\s*已编辑)\s*$", re.I)
_PAGE_INFO = re.compile(r"\s+[-–—(]\s*(?:Page|第)\s*\d+.*$", re.I)
_TERMINAL_NOISE = re.compile(r"\s+[-–—]\s+(?:\d+×\d+|zsh|bash|fish|-zsh|-bash|sh)\b.*$", re.I)
_SHELL_PROMPT = re.compile(r"^[\w.-]+@[\w.-]+:\s*")
_KNOWN_SITE_SUFFIXES = (
    "YouTube", "Wikipedia", "Stack Overflow", "GitHub", "Medium", "LinkedIn", "Reddit", "X", "Twitter",
    "Substack", "Google Docs", "Google Sheets", "Google Slides", "Notion", "Overleaf", "arXiv",
    "SSRN", "NBER", "JSTOR", "ScienceDirect", "Hacker News", "知乎", "哔哩哔哩", "bilibili", "微博",
    "小红书", "豆瓣", "ChatGPT", "Claude", "Perplexity", "Gemini", "Kaggle", "Hugging Face",
)
_SITE_SUFFIX = re.compile(r"^(?P<main>.+?)\s*(?:[-–—|·_]|::)\s*(?P<site>[^-–—|·_]{1,32})$")


@dataclass(frozen=True)
class Subject:
    key: str
    label: str
    kind: str  # focus | search
    category: str
    app: str
    title: str
    url: str
    domain: str


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "").casefold()
    text = re.sub(r"\s+", " ", text).strip(" -–—|·:;,.!?\"'“”‘’()[]")
    return text[:140]


def host_matches(host: str, pattern: str) -> bool:
    """'x.com' must not match 'dropbox.com'. Patterns ending in '.' ('google.')
    match a label run anywhere; full domains match exactly or as a parent."""
    host = host.lower().strip(".")
    if pattern.endswith("."):
        return f".{pattern}" in f".{host}."
    if re.search(r"\.[a-z]{2,}$", pattern):
        return host == pattern or host.endswith("." + pattern)
    return f".{pattern}." in f".{host}."


def is_browser(app: str) -> bool:
    lowered = app.casefold()
    return any(lowered == b or lowered.startswith(b + " ") or lowered.endswith(" " + b) for b in BROWSERS)


def categorize(app: str, domain: str = "") -> str:
    if domain:
        for category, hosts in DOMAIN_CATEGORIES:
            if any(host_matches(domain, h) for h in hosts):
                return category
    lowered = app.casefold()
    for category, names in APP_CATEGORIES:
        for name in names:
            if lowered == name or lowered.startswith(name + " ") or lowered.endswith(" " + name):
                return category
    return "browser" if is_browser(app) else "other"


def search_query(url: str, title: str = "") -> tuple[str, bool] | None:
    """Return (query, scholarly) if the url or title is a search results page."""
    if url:
        try:
            parts = urlsplit(url)
        except ValueError:
            parts = None
        if parts and parts.hostname:
            host = parts.hostname.lower()
            params = parse_qs(parts.query)
            for fragment, path, keys in SEARCH_ENGINES:
                if host_matches(host, fragment) and parts.path.startswith(path):
                    for key in keys:
                        value = (params.get(key) or [""])[0].strip()
                        if value:
                            scholarly = any(s in host for s in ("scholar.", "arxiv", "semanticscholar", "openalex"))
                            return value[:200], scholarly
    for pattern in TITLE_SEARCH_PATTERNS:
        match = pattern.match(title.strip())
        if match:
            return match.group("q").strip()[:200], "Scholar" in title or "学术" in title
    return None


def clean_title(title: str, app: str, domain: str = "") -> str:
    text = (title or "").replace("​", "").strip()
    text = _COUNTER.sub("", text)
    text = _BROWSER_SUFFIX.sub("", text)
    text = _MORE_PAGES.sub("", text)
    text = _PROFILE_SUFFIX.sub("", text)
    text = _BROWSER_SUFFIX.sub("", text)
    text = _EDITED.sub("", text)
    text = _PAGE_INFO.sub("", text)
    if app:
        app_suffix = re.compile(r"\s+[-–—]\s+" + re.escape(app) + r"\s*$", re.I)
        text = app_suffix.sub("", text)
    text = re.sub(r"\s+[-–—]\s+(?:Visual Studio Code|Microsoft Word|Word|Excel|PowerPoint|Pages|Numbers|Keynote)\s*$", "", text, flags=re.I)
    text = _TERMINAL_NOISE.sub("", text)
    text = _SHELL_PROMPT.sub("", text)
    match = _SITE_SUFFIX.match(text)
    if match:
        site = match.group("site").strip()
        site_token = re.sub(r"[^a-z0-9]", "", site.casefold())
        domain_token = re.sub(r"[^a-z0-9]", "", domain.split(".")[0]) if domain else ""
        if site in _KNOWN_SITE_SUFFIXES or (site_token and domain_token and (
            site_token in domain_token or domain_token in site_token
        )):
            text = match.group("main").strip()
    return text.strip()


def _code_project(title: str) -> str:
    """Editors title windows 'file — project'; the project is the stable subject."""
    parts = [p.strip() for p in re.split(r"\s+[-–—]\s+", title) if p.strip()]
    if len(parts) >= 2:
        return parts[1]
    return parts[0] if parts else ""


def describe(
    app: str,
    title: str,
    url: str,
    *,
    quiet_apps: list[str],
    quiet_domains: list[str],
    private_apps: list[str],
    capture_titles: bool = True,
    capture_urls: bool = True,
) -> Subject | None:
    """Return the subject for a sample, or None when it must not be recorded."""
    app = (app or "").strip()[:80]
    raw_title = (title or "").strip()
    raw_url = (url or "").strip() if capture_urls else ""
    domain = domain_of(raw_url)
    if is_quiet(app, domain, raw_title, quiet_apps, quiet_domains):
        return None

    category = categorize(app, domain)
    if is_private_app(app, private_apps) or category == "chat":
        # Conversations: record that time went here, never with whom or about what.
        return Subject(
            key=f"web:{domain}" if domain else f"app:{normalize(app)}",
            label=domain or app or "unknown app",
            kind="focus",
            category="chat",
            app=app,
            title="",
            url="",
            domain="",
        )
    if not capture_titles:
        return Subject(
            key=f"app:{normalize(app)}" + (f":{domain}" if domain else ""),
            label=domain or app or "unknown app",
            kind="focus",
            category=category,
            app=app,
            title="",
            url=clean_url(raw_url) if domain else "",
            domain=domain,
        )

    found = search_query(raw_url, raw_title)
    if found:
        query, scholarly = found
        query = redact(query)
        return Subject(
            key=f"search:{normalize(query)}",
            label=query,
            kind="search",
            category="reading" if scholarly else "search",
            app=app,
            title=redact(raw_title)[:300],
            url=clean_url(raw_url),
            domain=domain,
        )

    cleaned = redact(clean_title(raw_title, app, domain))
    if category == "code" and cleaned:
        project = _code_project(cleaned)
        key_part = normalize(project) or normalize(cleaned)
        return Subject(
            key=f"app:{normalize(app)}:{key_part}",
            label=project or cleaned,
            kind="focus",
            category=category,
            app=app,
            title=cleaned[:300],
            url="",
            domain="",
        )
    if domain:
        label = cleaned or domain
        return Subject(
            key=f"web:{domain}:{normalize(label)}",
            label=label,
            kind="focus",
            category=category,
            app=app,
            title=cleaned[:300],
            url=clean_url(raw_url),
            domain=domain,
        )
    label = cleaned or app or "unknown"
    return Subject(
        key=f"app:{normalize(app)}:{normalize(cleaned)}" if cleaned else f"app:{normalize(app)}",
        label=label,
        kind="focus",
        category=category,
        app=app,
        title=cleaned[:300],
        url="",
        domain="",
    )
