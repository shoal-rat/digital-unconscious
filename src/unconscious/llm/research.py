"""Letting Claude look things up while it dives.

With ``models.research`` on, the dream and the seabed give Claude Code a few tools:
web search, a reader for reference and scholarly pages, and a shell to download and
read papers. They work inside one temporary folder per errand, deleted afterwards.

The boundary, kept by Claude Code's own permission rules and the operating system's
sandbox rather than by asking the model nicely:

    web search                any query; it runs on Anthropic's side
    reading a page            only hosts on SHELF: encyclopedias, forums such as Reddit and Zhihu, social
                              and video sites, Chinese and English news, film and book sites, scholarly
                              indexes, preprint servers and publishers
    shell commands            sandboxed: write only inside the errand's folder, read nothing outside it,
                              reach only SHELF hosts, and never retried outside the sandbox
    files elsewhere           not readable at all (blockReadsOutsideWorkingDirectories)
    anything that would ask   refused: there is nobody to ask at night (--permission-prompts none)

The shelf is wide, so the crew can see how ideas live in popular culture as well as in
papers, but it holds only big platforms. A page or a PDF can carry instructions, and the
errand's prompt carries the person's day; a fetched page can steer Claude only to places
whose owners cannot read who asked for what. Personal hosting (OFF_SHELF) never gets on.
"""

from __future__ import annotations

import json
from typing import Any

TOOLS = "WebSearch,WebFetch,Read,Glob,Grep,Bash"

# Hosts a research errand may read from or download from. Each entry also allows its subdomains.
# Big platforms only: a site whose owner could read the request log of a URL Claude was talked
# into fetching would be a way out for the day's details. So no personal hosting (OFF_SHELF).
SHELF = (
    # encyclopedias, dictionaries, references
    "wikipedia.org", "wikimedia.org", "wiktionary.org", "wikiquote.org", "plato.stanford.edu", "iep.utm.edu",
    "britannica.com", "baike.baidu.com", "baike.sogou.com", "moegirl.org.cn", "fandom.com", "knowyourmeme.com",
    "urbandictionary.com", "merriam-webster.com", "dictionary.cambridge.org",
    # forums and question-and-answer
    "reddit.com", "redd.it", "redditmedia.com", "news.ycombinator.com", "stackexchange.com", "stackoverflow.com",
    "quora.com", "zhihu.com", "zhimg.com", "v2ex.com", "tieba.baidu.com", "zhidao.baidu.com", "douban.com",
    "github.com",
    # social and video
    "weibo.com", "weibo.cn", "bilibili.com", "xiaohongshu.com", "youtube.com", "x.com", "twitter.com",
    "medium.com", "substack.com", "mp.weixin.qq.com",
    # news and magazines, English
    "nytimes.com", "theguardian.com", "bbc.com", "bbc.co.uk", "reuters.com", "apnews.com", "economist.com",
    "ft.com", "wsj.com", "bloomberg.com", "washingtonpost.com", "theatlantic.com", "newyorker.com", "wired.com",
    "theverge.com", "arstechnica.com", "techcrunch.com", "vox.com", "npr.org", "aeon.co", "quantamagazine.org",
    "scientificamerican.com", "newscientist.com", "technologyreview.com", "hbr.org", "nautil.us", "time.com",
    "axios.com", "politico.com", "fortune.com", "forbes.com", "cnn.com", "cnbc.com",
    # news and magazines, Chinese
    "thepaper.cn", "caixin.com", "jiemian.com", "yicai.com", "infzm.com", "bjnews.com.cn", "people.com.cn",
    "xinhuanet.com", "news.cn", "chinadaily.com.cn", "cctv.com", "sina.com.cn", "163.com", "sohu.com", "ifeng.com",
    "news.qq.com", "zaobao.com", "ftchinese.com", "theinitium.com", "huxiu.com", "36kr.com", "sspai.com",
    "geekpark.net", "ithome.com", "guokr.com", "jiqizhixin.com", "qbitai.com",
    # film, books, music, games
    "imdb.com", "rottentomatoes.com", "metacritic.com", "letterboxd.com", "goodreads.com", "steampowered.com",
    "maoyan.com", "music.163.com", "pitchfork.com", "billboard.com",
    # indexes and identifiers
    "openalex.org", "doi.org", "semanticscholar.org", "crossref.org", "unpaywall.org", "core.ac.uk",
    "scholar.archive.org", "europepmc.org", "ncbi.nlm.nih.gov", "dblp.org", "cnki.net", "wanfangdata.com.cn",
    # preprint and working-paper servers
    "arxiv.org", "biorxiv.org", "medrxiv.org", "osf.io", "ssrn.com", "nber.org", "repec.org", "zenodo.org",
    "hal.science", "openreview.net", "aclanthology.org", "proceedings.mlr.press", "neurips.cc", "jmlr.org",
    "chinaxiv.org",
    # publishers and societies
    "aeaweb.org", "econometricsociety.org", "informs.org", "uchicago.edu", "oup.com", "cambridge.org",
    "springer.com", "nature.com", "wiley.com", "sciencedirect.com", "elsevier.com", "tandfonline.com",
    "sagepub.com", "jstor.org", "science.org", "pnas.org", "plos.org", "frontiersin.org", "mdpi.com",
    "cell.com", "bmj.com", "thelancet.com", "nejm.org", "annualreviews.org", "acm.org", "ieee.org",
    "apa.org", "direct.mit.edu", "royalsocietypublishing.org", "aisnet.org",
)

# Never on the shelf: anyone can run a site here and read who fetched what.
OFF_SHELF = (
    "github.io", "githubusercontent.com", "pages.dev", "workers.dev", "vercel.app",
    "netlify.app", "herokuapp.com", "glitch.me", "repl.co", "ngrok.io", "ngrok-free.app", "trycloudflare.com",
    "webhook.site", "requestbin.com", "pipedream.net", "pastebin.com", "blogspot.com", "wordpress.com",
)

# What one research errand may spend at most, as Claude Code counts it.
BUDGET_USD = {"dream": 6.0, "dive": 4.0}
TIMEOUT_SECONDS = 1200


def domains() -> list[str]:
    out: list[str] = []
    for host in SHELF:
        out += [host, f"*.{host}"]
    return out


def off_shelf(host: str) -> bool:
    return any(host == bad or host.endswith("." + bad) for bad in OFF_SHELF)


def claude_settings() -> dict[str, Any]:
    allowed = domains()
    return {
        "permissions": {
            "allow": ["WebSearch", *(f"WebFetch(domain:{host})" for host in allowed)],
            "blockReadsOutsideWorkingDirectories": True,
        },
        "sandbox": {
            "enabled": True,
            "failIfUnavailable": True,
            "autoAllowBashIfSandboxed": True,
            "allowUnsandboxedCommands": False,
            "network": {"allowedDomains": allowed, "strictAllowlist": True},
        },
    }


def claude_flags(role: str) -> list[str]:
    return [
        "--tools", TOOLS,
        "--settings", json.dumps(claude_settings(), separators=(",", ":")),
        "--permission-prompts", "none",
        "--max-budget-usd", str(BUDGET_USD.get(role, 3.0)),
    ]
