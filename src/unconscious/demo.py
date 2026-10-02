"""Three weeks of sample memory for a fictional researcher, for `dun demo`.

Traces are generated; signals are computed from them by the real code; the
dreams are hand-written so the dashboard shows what good output looks like
before any model is connected. All cited papers are real, well-known works.
"""

from __future__ import annotations

import os
import random
import shutil
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

PERSONA = {
    "en": (
        "PhD candidate in marketing and behavioural economics. My dissertation studies subscription pricing and "
        "churn with a firm-level panel. I want empirical ideas I could test with data I have or can get."
    ),
    "zh": "市场营销与行为经济学方向的博士生。论文用企业面板数据研究订阅定价与用户流失。想要能用现有或可获得的数据检验的实证研究点子。",
}
FOCUS = {"en": ["behavioural economics", "marketing", "pricing"], "zh": ["行为经济学", "市场营销", "定价"]}


@dataclass
class Spec:
    key: str
    name: dict[str, str]
    gist: dict[str, str]
    keywords: list[str]
    pages: list[tuple[str, str, str]]  # app, title, url
    searches: list[str] = field(default_factory=list)
    state: str = "active"


SPECS = [
    Spec(
        "pricing",
        {"en": "SaaS pricing psychology", "zh": "SaaS 定价心理"},
        {"en": "How plan names, decoy tiers and table layout shape what people buy and whether they stay.",
         "zh": "套餐命名、诱饵档位和价格表布局如何影响人们买什么、留不留下。"},
        ["pricing", "decoy", "plan naming", "saas", "price table"],
        [
            ("Google Chrome", "Pricing – Linear", "https://linear.app/pricing"),
            ("Google Chrome", "Pricing | Notion", "https://www.notion.so/pricing"),
            ("Google Chrome", "Plans and pricing - Figma", "https://www.figma.com/pricing"),
            ("Preview", "Huber_Payne_Puto_1982_asymmetric_dominance.pdf", ""),
            ("Google Chrome", "Shrouded Attributes, Consumer Myopia, and Information Suppression | The Quarterly Journal of Economics", "https://academic.oup.com/qje/article/121/2/505/1884014"),
        ],
        ["decoy effect saas pricing page", "plan naming cognitive load", "three tier pricing middle option share"],
    ),
    Spec(
        "churn",
        {"en": "Churn panel models", "zh": "流失面板模型"},
        {"en": "The dissertation core: survival models of subscription churn on a firm panel.",
         "zh": "论文主线：在企业面板上用生存模型研究订阅流失。"},
        ["churn", "survival", "hazard", "panel", "cox"],
        [
            ("Visual Studio Code", "hazard_model.py — churn-panel — Visual Studio Code", ""),
            ("Visual Studio Code", "cohorts.py — churn-panel — Visual Studio Code", ""),
            ("Stata", "churn_panel.do", ""),
            ("Google Chrome", "Time varying survival regression — lifelines documentation", "https://lifelines.readthedocs.io/en/latest/Time%20varying%20survival%20regression.html"),
            ("Terminal", "zwk@mac: ~/churn-panel — zsh — 120×32", ""),
        ],
        ["cox model time varying covariates", "frailty model subscription churn"],
    ),
    Spec(
        "bikes",
        {"en": "Bike lanes and commuting", "zh": "自行车道与通勤"},
        {"en": "Protected bike lanes, who starts cycling, and open trip data.",
         "zh": "受保护自行车道、谁开始骑车，以及公开的骑行数据。"},
        ["bike lanes", "cycling", "commuting", "ridership", "divvy"],
        [
            ("Google Chrome", "Bike Lanes and Routes - City of Chicago", "https://www.chicago.gov/city/en/depts/cdot/provdrs/bike.html"),
            ("Google Chrome", "Divvy System Data | Divvy Bikes", "https://divvybikes.com/system-data"),
            ("Google Chrome", "Protected bike lanes and who rides - Streetsblog USA", "https://usa.streetsblog.org/"),
        ],
        ["bike lane ridership before after", "divvy trips by station 2023"],
    ),
    Spec(
        "sleep",
        {"en": "Sleep and decision quality", "zh": "睡眠与决策质量"},
        {"en": "How sleep loss and late hours change risk-taking and impulsive choice.",
         "zh": "睡眠不足和深夜如何改变冒险与冲动决策。"},
        ["sleep", "fatigue", "impulsivity", "risk taking", "circadian"],
        [
            ("Google Chrome", "Sleep Deprivation Biases the Neural Mechanisms Underlying Economic Preferences | Journal of Neuroscience", "https://www.jneurosci.org/content/31/10/3712"),
            ("Google Chrome", "Effects of sleep deprivation on cognition - ScienceDirect", "https://www.sciencedirect.com/science/article/pii/B9780444537027000075"),
        ],
        ["sleep deprivation risk taking", "time of day impulse purchases"],
    ),
    Spec(
        "llmeval",
        {"en": "Evaluating language models", "zh": "语言模型评测"},
        {"en": "Benchmarks, contamination, and whether an LLM judge can be trusted to code survey answers.",
         "zh": "基准、数据污染，以及能否信任 LLM 评委来给问卷开放题编码。"},
        ["llm", "evaluation", "benchmark", "llm judge", "contamination"],
        [
            ("Google Chrome", "[2306.05685] Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena", "https://arxiv.org/abs/2306.05685"),
            ("Google Chrome", "[2211.09110] Holistic Evaluation of Language Models", "https://arxiv.org/abs/2211.09110"),
            ("Google Chrome", "EleutherAI/lm-evaluation-harness · GitHub", "https://github.com/EleutherAI/lm-evaluation-harness"),
            ("ChatGPT", "Coding open-ended survey answers", ""),
        ],
        ["llm judge position bias", "benchmark contamination detection"],
    ),
    Spec(
        "menu",
        {"en": "Menu design and choice", "zh": "菜单设计与选择"},
        {"en": "Restaurant menu engineering: anchors, decoy dishes and the layout of choices.",
         "zh": "餐厅菜单工程：锚定、诱饵菜和选项布局。"},
        ["menu", "restaurant", "anchoring", "choice overload", "menu engineering"],
        [
            ("Google Chrome", "The secret psychology of restaurant menus - BBC", "https://www.bbc.com/future/"),
            ("Google Chrome", "Can There Ever Be Too Many Options? A Meta-Analytic Review of Choice Overload | Journal of Consumer Research", "https://academic.oup.com/jcr/article/37/3/409/1827431"),
        ],
        ["menu engineering decoy dish", "menu anchoring field experiment"],
    ),
    Spec(
        "jobs",
        {"en": "Academic job market", "zh": "学术求职"},
        {"en": "Getting ready for the job market: the talk, the packet, the list.",
         "zh": "准备学术求职：求职报告、材料和投递清单。"},
        ["job market", "job talk", "applications"],
        [
            ("Google Chrome", "EconJobMarket.org", "https://econjobmarket.org/"),
            ("Keynote", "job_talk_v3", ""),
        ],
        ["how to structure a job market talk"],
    ),
    Spec(
        "flat",
        {"en": "Apartment hunting", "zh": "找房"},
        {"en": "Looking for a flat closer to campus.", "zh": "在学校附近找房子。"},
        ["apartment", "rent", "lease"],
        [("Google Chrome", "Apartments for rent near Hyde Park - Zillow", "https://www.zillow.com/")],
        ["1br hyde park chicago"],
        state="muted",
    ),
]

NOISE = [
    ("Slack", "general", ""),
    ("Mail", "Inbox", ""),
    ("Google Chrome", "YouTube", "https://www.youtube.com/"),
    ("Finder", "Downloads", ""),
]

JOTS = {
    "en": {
        0: [("sleep", "Do people sign up for pricier plans late at night? Our signup timestamps could show that."),
            ("menu", "BBC menu piece: the absurdly expensive dish is literally a decoy.")],
        3: [("pricing", "Plan names like 'Pro' vs 'Business' — does anyone know what they mean?")],
        9: [("llmeval", "If an LLM codes our survey answers, how would a referee check it?")],
    },
    "zh": {
        0: [("sleep", "深夜注册的人会不会更容易选贵的套餐？我们的注册时间戳也许能看出来。"),
            ("menu", "BBC 那篇讲菜单：那道贵得离谱的菜就是诱饵。")],
        3: [("pricing", "“Pro”和“Business”这种套餐名，用户真的知道区别吗？")],
        9: [("llmeval", "如果用 LLM 给问卷开放题编码，审稿人要怎么核验？")],
    },
}

READINGS = {
    5: ("pricing", "Search, Obfuscation, and Price Elasticities on the Internet",
        "Notes on Ellison & Ellison (Econometrica, 2009): when search is cheap, demand for near-identical products "
        "is extremely price-elastic, and sellers respond by making prices harder to compare — add-ons, confusing "
        "bundles, and deliberately inferior base products."),
}


def _minutes_for(key: str, offset: int, weekday: int, rng: random.Random) -> int:
    weekend = weekday >= 5
    if key == "pricing":
        if offset in {0, 5}:
            return 38 if offset == 0 else 44
        return rng.randint(15, 50) if rng.random() < (0.3 if weekend else 0.7) else 0
    if key == "churn":
        if offset == 0:
            return 125
        if weekend:
            return rng.randint(25, 60) if rng.random() < 0.2 else 0
        return rng.randint(70, 170) if rng.random() < 0.9 else 0
    if key == "bikes":
        return rng.randint(2, 7) if offset in {0, 1, 3, 4, 6, 8, 10, 12, 15, 18} else 0
    if key == "sleep":
        if offset == 0:
            return 26
        return rng.randint(15, 35) if offset in {12, 13, 14, 16, 17, 19} else 0
    if key == "llmeval":
        if offset == 0:
            return 95
        return rng.randint(8, 18) if offset in {2, 5, 9, 11, 15} else 0
    if key == "menu":
        return 35 if offset == 0 else 0
    if key == "jobs":
        return rng.randint(10, 30) if offset and rng.random() < 0.4 else 0
    if key == "flat":
        return rng.randint(10, 25) if rng.random() < 0.3 else 0
    return 0


def _split(total: int, rng: random.Random) -> list[int]:
    pieces, left = [], total
    while left > 0:
        piece = min(left, rng.randint(2, 9))
        pieces.append(piece)
        left -= piece
    return pieces


def seed_demo(home: Path, language: str = "en", reset: bool = True) -> dict[str, Any]:
    from unconscious.app import App
    from unconscious.mind.signals import compute_signals
    from unconscious.sense.subjects import describe

    home = Path(home)
    if reset and home.exists():
        shutil.rmtree(home)
    previous = os.environ.get("DUN_HOME")
    os.environ["DUN_HOME"] = str(home)
    try:
        app = App(home)
    finally:
        if previous is None:
            os.environ.pop("DUN_HOME", None)
        else:
            os.environ["DUN_HOME"] = previous
    app.update_settings({
        "you": {"name": "Alex" if language == "en" else "", "persona": PERSONA[language], "focus": FOCUS[language], "language": language},
        "dream": {"auto": False},
    })
    store, settings = app.store, app.settings
    rng = random.Random(7)
    now = datetime.now().astimezone()
    today = now.date()
    thread_ids: dict[str, int] = {}
    first_days: dict[str, str] = {}
    trace_total = 0
    per_day: dict[str, dict[str, dict[str, Any]]] = {}

    for offset in range(20, -1, -1):
        day = today - timedelta(days=offset)
        day_s = day.isoformat()
        blocks: list[tuple[str | None, int]] = []
        for spec in SPECS:
            minutes = _minutes_for(spec.key, offset, day.weekday(), rng)
            if minutes:
                blocks.append((spec.key, minutes))
        if day.weekday() < 5:
            blocks.append((None, rng.randint(20, 40)))
        blocks.append((None, rng.randint(5, 15)))
        rng.shuffle(blocks)
        blocks.sort(key=lambda b: 0 if b[0] == "churn" else 1)  # deep work in the morning
        needed = sum(m for _, m in blocks) + 4 * len(blocks)
        start = datetime.combine(day, datetime.min.time()).astimezone().replace(hour=8, minute=40 + rng.randint(0, 15))
        if offset == 0:
            latest_end = now - timedelta(minutes=8)
            start = min(start, latest_end - timedelta(minutes=needed))
            start = max(start, datetime.combine(day, datetime.min.time()).astimezone() + timedelta(minutes=5))
        clock = start
        jots_today = dict(JOTS[language]).get(offset, [])
        for key, minutes in blocks:
            spec = next((s for s in SPECS if s.key == key), None)
            for piece in _split(minutes, rng):
                if spec is None:
                    app_name, title, url = rng.choice(NOISE)
                else:
                    if spec.searches and rng.random() < 0.22:
                        query = rng.choice(spec.searches)
                        app_name, title, url = "Google Chrome", f"{query} - Google Search", "https://www.google.com/search?q=" + query.replace(" ", "+")
                    else:
                        app_name, title, url = rng.choice(spec.pages)
                subject = describe(app_name, title, url, quiet_apps=settings.sense.quiet_apps,
                                   quiet_domains=settings.sense.quiet_domains, private_apps=settings.sense.private_apps)
                if subject is None:
                    continue
                seconds = piece * 60 - rng.randint(0, 40)
                end = clock + timedelta(seconds=seconds)
                store.add_trace(
                    day=day_s, started_at=clock.isoformat(timespec="seconds"), ended_at=end.isoformat(timespec="seconds"),
                    seconds=float(seconds), kind=subject.kind, source="sensor", app=subject.app, category=subject.category,
                    title=subject.title, url=subject.url, domain=subject.domain, subject_key=subject.key, subject=subject.label,
                )
                trace_total += 1
                if spec is not None:
                    bucket = per_day.setdefault(day_s, {}).setdefault(spec.key, {"seconds": 0.0, "visits": 0, "subjects": {}})
                    bucket["seconds"] += seconds
                    bucket["visits"] += 1
                    bucket["subjects"][subject.key] = subject.label
                clock = end + timedelta(seconds=rng.randint(10, 90))
            for jot_key, text in [j for j in jots_today if j[0] == key]:
                _add_note(store, day_s, clock, "jot", text, per_day, jot_key)
                trace_total += 1
            if offset in READINGS and READINGS[offset][0] == key:
                _, title, body = READINGS[offset]
                _add_note(store, day_s, clock, "reading", body, per_day, key, title=title)
                trace_total += 1
            clock += timedelta(minutes=rng.randint(2, 6))
        for key in per_day.get(day_s, {}):
            first_days.setdefault(key, day_s)

    for spec in SPECS:
        if spec.key not in first_days:
            continue
        tid = store.create_thread(spec.name[language], spec.gist[language], spec.keywords, first_days[spec.key])
        if spec.state != "active":
            store.update_thread(tid, state=spec.state)
        thread_ids[spec.key] = tid
    for day_s, buckets in per_day.items():
        rows, mapping, topics = [], {}, []
        for key, bucket in buckets.items():
            tid = thread_ids[key]
            labels = list(bucket["subjects"].values())
            rows.append({"thread_id": tid, "seconds": round(bucket["seconds"], 1), "visits": bucket["visits"],
                         "subjects": labels, "note": ""})
            for subject_key in bucket["subjects"]:
                mapping[subject_key] = tid
            spec = next(s for s in SPECS if s.key == key)
            topics.append({"label": spec.name[language], "gist": spec.gist[language], "thread_id": tid,
                           "keywords": spec.keywords[:4], "subject_keys": list(bucket["subjects"]),
                           "seconds": round(bucket["seconds"]), "visits": bucket["visits"]})
        store.replace_day_assignments(day_s, rows, mapping)
        store.save_digest(day_s, {"topics": sorted(topics, key=lambda t: -t["seconds"])}, store.trace_count(day_s), "demo")

    dreams = _write_dreams(app, today, thread_ids, language, compute_signals)
    return {"days": 21, "traces": trace_total, "threads": len(thread_ids), "dreams": dreams, "home": str(home)}


def _add_note(store, day_s: str, at: datetime, kind: str, text: str, per_day, key: str, title: str = "") -> None:
    import hashlib

    digest = hashlib.sha1(text.encode()).hexdigest()[:12]
    subject_key = f"{'jot' if kind == 'jot' else 'doc'}:{digest}"
    label = title or text[:140]
    store.add_trace(
        day=day_s, started_at=at.isoformat(timespec="seconds"), ended_at=at.isoformat(timespec="seconds"), seconds=0.0,
        kind=kind, source="jot" if kind == "jot" else "feed", category="note" if kind == "jot" else "reading",
        title=title, body=text, subject_key=subject_key, subject=label,
    )
    bucket = per_day.setdefault(day_s, {}).setdefault(key, {"seconds": 0.0, "visits": 0, "subjects": {}})
    bucket["visits"] += 1
    bucket["subjects"][subject_key] = label


# ------------------------------------------------------------------- dreams

DREAMS: dict[str, list[dict[str, Any]]] = {
    "en": [
        {
            "offset": 0,
            "title": "Late-night choices, priced in daylight",
            "reflection": "You spent the morning inside the churn panel, but your attention kept slipping sideways: to a piece on "
                          "restaurant menus, to an LLM-evaluation paper you opened three times, and — for the first time in almost "
                          "two weeks — back to sleep and impulsive choice. Pricing pages sat open most of the day, more returned to than read.",
            "undercurrent": "Is the person choosing a plan at 11 p.m. the same decision-maker your models assume?",
            "sparks": [
                {
                    "mechanism": "collision", "threads": ["sleep", "pricing", "churn"], "score": 86,
                    "title": "Do late-night signups pick the decoy?",
                    "question": "In your firm panel, are plans chosen between 22:00 and 03:00 more likely to be the anchored middle tier — and do those accounts churn faster?",
                    "insight": "Sleep and impulsivity resurfaced today on the same afternoon you were reading about decoy tiers. Your panel already "
                               "stores signup timestamps, a free proxy for decision fatigue that pricing studies rarely use. If late choices skew "
                               "toward the anchored tier and churn sooner, choice architecture affects retention, not just conversion.",
                    "first_step": "Tabulate plan choice by local signup hour for one cohort and plot each tier's share by hour.",
                    "kill": "If the tier mix is flat across hours after controlling for country and weekday, drop it.",
                    "field": "behavioural economics",
                    "objection": "Hour of day is confounded with who signs up: late-night signups may be students or other time zones.",
                    "scores": {"grounded": 5, "sharp": 5, "fresh": 4, "doable": 5, "fit": 5, "generic": False},
                    "evidence": ["jot:sleep", "search:sleep", "page:pricing", "page:churn"],
                    "search_terms": ["time of day consumer choice fatigue", "decoy effect decision fatigue", "late night purchases impulsivity"],
                },
                {
                    "mechanism": "orbit", "threads": ["bikes"], "score": 78,
                    "title": "The bike-lane question you keep not asking",
                    "question": "When a protected lane opens on a corridor, does nearby bike-share ridership grow from new riders or from riders switching routes?",
                    "insight": "You have visited bike-lane pages on most of the last two weeks and never stayed more than a few minutes. The pages "
                               "you return to are about ridership, not safety, and the trip data you glance at is open and station-level. "
                               "That is an event study waiting for a weekend.",
                    "first_step": "Pick one lane that opened in 2023 and pull trips for stations within 400 m, six months before and after.",
                    "kill": "If ridership near the lane rises no more than at matched stations elsewhere, the lane did not change behaviour.",
                    "field": "urban economics",
                    "objection": "Station-level trips cannot tell new riders from switchers without rider identifiers.",
                    "scores": {"grounded": 4, "sharp": 4, "fresh": 4, "doable": 4, "fit": 3, "generic": False},
                    "evidence": ["page:bikes", "search:bikes"],
                    "search_terms": ["protected bike lane ridership event study", "bike share station demand infrastructure"],
                },
                {
                    "mechanism": "seed", "threads": ["menu", "pricing"], "score": 72,
                    "title": "Menus are pricing pages with plates",
                    "question": "Does an extreme 'anchor dish' shift orders the way an enterprise tier shifts SaaS plan choice — and is the effect larger on longer menus?",
                    "insight": "You found menu engineering today and read it like a pricing researcher: your searches were about decoys and anchors, "
                               "not food. Menu field experiments are cheap and well documented, which could give your pricing chapter a clean "
                               "second setting for the same mechanism.",
                    "first_step": "Find two published menu field experiments and note whether their anchoring effects are comparable to SaaS decoy estimates.",
                    "kill": "If menu anchoring effects differ by an order of magnitude, the analogy will not carry a paper.",
                    "field": "marketing",
                    "objection": "Restaurant orders are repeated, social and sensory; plan choice is rare and solitary.",
                    "scores": {"grounded": 4, "sharp": 4, "fresh": 3, "doable": 4, "fit": 4, "generic": False},
                    "evidence": ["jot:menu", "page:menu", "search:menu"],
                    "search_terms": ["menu anchoring field experiment", "decoy dish menu design"],
                },
            ],
        },
        {
            "offset": 2,
            "title": "Benchmarks you do not quite trust",
            "reflection": "A short day in the panel and a long detour through LLM evaluation: a paper on LLM judges, the evaluation harness on GitHub, "
                          "and a conversation about coding survey answers. The detours were not random; each one was about verification.",
            "undercurrent": "What would it take for you to believe a number a model produced?",
            "sparks": [
                {
                    "mechanism": "gap", "threads": ["llmeval", "churn"], "score": 74, "status": "kept",
                    "title": "A referee-proof audit for LLM-coded survey answers",
                    "question": "If an LLM codes your exit-survey free text, how much do churn-reason shares move when you swap the model or reorder the categories?",
                    "insight": "You read about position bias in LLM judges on the same day you were asking whether a model could code your exit survey. "
                               "Reporting a sensitivity table is cheap, and it pre-empts the first question a referee will ask.",
                    "first_step": "Code 200 answers twice with shuffled category order and compute agreement.",
                    "kill": "If agreement is above 0.9 across orders and models, the audit is a footnote, not a contribution.",
                    "field": "research methods",
                    "objection": "Agreement between models is not accuracy; you still need a human-coded subset.",
                    "scores": {"grounded": 4, "sharp": 4, "fresh": 3, "doable": 5, "fit": 4, "generic": False},
                    "evidence": ["page:llmeval", "search:llmeval"],
                    "search_terms": ["llm judge position bias", "llm annotation reliability survey coding"],
                },
                {
                    "mechanism": "surge", "threads": ["llmeval"], "score": 51, "status": "dismissed", "reason": "generic",
                    "title": "Use an LLM to summarise customer feedback",
                    "question": "Can an LLM summarise churned users' feedback into themes?",
                    "insight": "You spent time on LLM tools; summarising feedback could save time.",
                    "first_step": "Paste a sample of feedback into a model.",
                    "kill": "If the themes are obvious, stop.",
                    "field": "marketing",
                    "objection": "This is a tool, not a question, and anyone could have suggested it.",
                    "scores": {"grounded": 2, "sharp": 2, "fresh": 1, "doable": 5, "fit": 2, "generic": True},
                    "evidence": ["page:llmeval"],
                    "search_terms": [],
                },
            ],
        },
        {
            "offset": 5,
            "title": "The middle tier, again",
            "reflection": "You came back to three-tier pricing pages four times today and fed in your notes on obfuscation. The searches were "
                          "about plan names, not prices: you seem less interested in how much things cost than in whether people can tell the options apart.",
            "undercurrent": "Do people churn from the price, or from never having understood what they bought?",
            "sparks": [
                {
                    "mechanism": "orbit", "threads": ["pricing", "churn"], "score": 82, "status": "pursuing",
                    "title": "Plan names as cognitive load",
                    "question": "Do firms whose tier names are less descriptive (\"Pro\", \"Plus\") see higher early churn than firms with descriptive names (\"Team\", \"Up to 10 seats\")?",
                    "insight": "Your searches circle plan naming rather than price levels, and your notes on obfuscation describe exactly the mechanism: "
                               "making options hard to compare. Tier names are public, so they can be coded for every firm in your panel.",
                    "first_step": "Code tier names for 30 firms as descriptive or abstract and compare 90-day churn.",
                    "kill": "If abstract names do not predict churn once price dispersion is controlled, it is a branding story, not a choice story.",
                    "field": "marketing",
                    "objection": "Firms that use abstract names may differ in many other ways, such as product complexity.",
                    "scores": {"grounded": 5, "sharp": 4, "fresh": 4, "doable": 4, "fit": 5, "generic": False},
                    "evidence": ["page:pricing", "search:pricing", "reading:pricing"],
                    "search_terms": ["plan name complexity churn", "price obfuscation consumer confusion", "choice overload subscription"],
                    "dive": True,
                },
            ],
        },
    ],
}

DREAMS["zh"] = [
    {
        **DREAMS["en"][0],
        "title": "深夜的选择，白天的定价",
        "reflection": "上午你一直在流失面板里，但注意力不断往旁边滑：一篇讲餐厅菜单的文章、一篇你打开了三次的 LLM 评测论文，"
                      "以及——近两周来第一次——又回到了睡眠与冲动决策。定价页面开了一整天，更多是被反复切回，而不是被读完。",
        "undercurrent": "深夜 11 点选套餐的那个人，和你模型里假设的决策者是同一个人吗？",
        "sparks": [
            {**DREAMS["en"][0]["sparks"][0],
             "title": "深夜注册的人更容易选中诱饵档吗？",
             "question": "在你的企业面板里，22:00 到凌晨 3:00 之间选择的套餐，是否更可能是被锚定的中间档？这些账户是否流失得更快？",
             "insight": "今天下午你在读诱饵档位，同一个下午睡眠与冲动又浮现了。你的面板本来就记录了注册时间戳——这是定价研究很少用到的、"
                        "免费的决策疲劳代理变量。如果深夜的选择偏向锚定档且更早流失，那么选择架构影响的就不只是转化，还有留存。",
             "first_step": "取一个注册批次，按当地注册小时统计套餐选择，画出各档位随小时变化的份额。",
             "kill": "如果控制国家和星期后，各小时的档位结构没有差别，就放弃。",
             "field": "行为经济学",
             "objection": "注册时段与注册人群相关：深夜注册的可能是学生或其他时区的用户。"},
            {**DREAMS["en"][0]["sparks"][1],
             "title": "那个你一直没问出口的自行车道问题",
             "question": "某条走廊开通受保护自行车道后，附近共享单车骑行量的增长，来自新骑手还是改道的老骑手？",
             "insight": "过去两周你大多数日子都打开过自行车道相关的页面，却从没停留超过几分钟。你反复回去看的是骑行量，而不是安全；"
                        "你瞥过的骑行数据是公开的、到站点级别的。这是一个等着周末去做的事件研究。",
             "first_step": "挑一条 2023 年开通的车道，取 400 米内站点在开通前后各六个月的骑行记录。",
             "kill": "如果车道附近的骑行增长不比其他匹配站点多，说明车道没有改变行为。",
             "field": "城市经济学",
             "objection": "没有骑手 ID，站点级数据分不清新骑手和改道者。"},
            {**DREAMS["en"][0]["sparks"][2],
             "title": "菜单就是带盘子的价格页",
             "question": "一道贵得离谱的“锚定菜”对点单的影响，是否像企业版档位影响 SaaS 套餐选择一样？菜单越长效应越大吗？",
             "insight": "今天你第一次接触菜单工程，而且是用定价研究者的眼光在读：你搜的是诱饵和锚定，不是食物。"
                        "菜单田野实验便宜且文献丰富，可以为你定价章节里的同一机制提供一个干净的第二场景。",
             "first_step": "找两篇已发表的菜单田野实验，记下它们的锚定效应量是否与 SaaS 诱饵估计可比。",
             "kill": "如果菜单锚定效应相差一个数量级，这个类比撑不起一篇论文。",
             "field": "市场营销",
             "objection": "点餐是重复的、社交的、感官的；选套餐是罕见且独自完成的。"},
        ],
    },
    {
        **DREAMS["en"][1],
        "title": "你并不完全信任的基准",
        "reflection": "在面板里待得不长，却在 LLM 评测上绕了很久：一篇讲 LLM 评委的论文、GitHub 上的评测框架、一次关于给问卷编码的对话。"
                      "这些绕路并不随机，每一次都关于“核验”。",
        "undercurrent": "要怎样你才会相信一个由模型产出的数字？",
        "sparks": [
            {**DREAMS["en"][1]["sparks"][0],
             "title": "让审稿人挑不出毛病的 LLM 编码审计",
             "question": "如果用 LLM 给退订问卷的开放回答编码，换一个模型或打乱类别顺序，各流失原因的占比会变化多少？",
             "insight": "你在读 LLM 评委的位置偏差的同一天，也在问模型能不能给你的退订问卷编码。报告一张敏感性表很便宜，"
                        "却能提前回答审稿人最先会问的问题。",
             "first_step": "用两种打乱的类别顺序给 200 条回答各编码一次，计算一致性。",
             "kill": "如果不同顺序与模型之间的一致性都高于 0.9，这个审计只配做脚注。",
             "field": "研究方法",
             "objection": "模型之间一致不等于准确，仍需要一个人工编码的子样本。"},
            {**DREAMS["en"][1]["sparks"][1],
             "title": "用 LLM 总结用户反馈",
             "question": "LLM 能否把流失用户的反馈总结成主题？",
             "insight": "你花了些时间在 LLM 工具上；总结反馈可以省时间。",
             "first_step": "把一批反馈粘贴进模型。",
             "kill": "如果主题显而易见，就停下。",
             "field": "市场营销",
             "objection": "这是一个工具，不是一个问题，谁都能提出来。"},
        ],
    },
    {
        **DREAMS["en"][2],
        "title": "又是中间那一档",
        "reflection": "今天你四次回到三档定价页面，还导入了关于价格混淆的笔记。你搜索的是套餐名称而不是价格：比起东西多少钱，"
                      "你似乎更在意人们能不能分清这些选项。",
        "undercurrent": "人们流失，是因为价格，还是因为从来没弄明白自己买了什么？",
        "sparks": [
            {**DREAMS["en"][2]["sparks"][0],
             "title": "套餐名称即认知负荷",
             "question": "套餐名称不具描述性（“Pro”“Plus”）的公司，早期流失是否高于名称具描述性（“团队版”“最多 10 个席位”）的公司？",
             "insight": "你的搜索绕着套餐命名转，而不是价格水平；你关于价格混淆的笔记描述的正是这个机制：让选项难以比较。"
                        "套餐名称是公开的，可以为面板里每家公司编码。",
             "first_step": "给 30 家公司的套餐名称编码为“描述性/抽象”，比较 90 天流失率。",
             "kill": "如果控制价格离散度后抽象名称不能预测流失，那这是品牌故事，不是选择故事。",
             "field": "市场营销",
             "objection": "使用抽象名称的公司可能在产品复杂度等许多方面不同。"},
        ],
    },
]

DIVE_PAPERS = [
    {"title": "Adding Asymmetrically Dominated Alternatives: Violations of Regularity and the Similarity Hypothesis",
     "year": 1982, "venue": "Journal of Consumer Research", "authors": ["Joel Huber", "John W. Payne", "Christopher Puto"],
     "cited_by": 0, "url": "https://doi.org/10.1086/208899", "abstract": ""},
    {"title": "When choice is demotivating: Can one desire too much of a good thing?", "year": 2000,
     "venue": "Journal of Personality and Social Psychology", "authors": ["Sheena S. Iyengar", "Mark R. Lepper"],
     "cited_by": 0, "url": "https://doi.org/10.1037/0022-3514.79.6.995", "abstract": ""},
    {"title": "Can There Ever Be Too Many Options? A Meta-Analytic Review of Choice Overload", "year": 2010,
     "venue": "Journal of Consumer Research", "authors": ["Benjamin Scheibehenne", "Rainer Greifeneder", "Peter M. Todd"],
     "cited_by": 0, "url": "https://doi.org/10.1086/651235", "abstract": ""},
    {"title": "Choice overload: A conceptual review and meta-analysis", "year": 2015,
     "venue": "Journal of Consumer Psychology", "authors": ["Alexander Chernev", "Ulf Böckenholt", "Joseph Goodman"],
     "cited_by": 0, "url": "https://doi.org/10.1016/j.jcps.2014.08.002", "abstract": ""},
    {"title": "Search, Obfuscation, and Price Elasticities on the Internet", "year": 2009, "venue": "Econometrica",
     "authors": ["Glenn Ellison", "Sara Fisher Ellison"], "cited_by": 0, "url": "https://doi.org/10.3982/ECTA5708", "abstract": ""},
    {"title": "Shrouded Attributes, Consumer Myopia, and Information Suppression in Competitive Markets", "year": 2006,
     "venue": "The Quarterly Journal of Economics", "authors": ["Xavier Gabaix", "David Laibson"], "cited_by": 0,
     "url": "https://doi.org/10.1162/qjec.2006.121.2.505", "abstract": ""},
    {"title": "Paying Not to Go to the Gym", "year": 2006, "venue": "American Economic Review",
     "authors": ["Stefano DellaVigna", "Ulrike Malmendier"], "cited_by": 0, "url": "https://doi.org/10.1257/aer.96.3.694", "abstract": ""},
    {"title": "Retention Futility: Targeting High-Risk Customers Might Be Ineffective", "year": 2018,
     "venue": "Journal of Marketing Research", "authors": ["Eva Ascarza"], "cited_by": 0,
     "url": "https://doi.org/10.1509/jmr.16.0163", "abstract": ""},
]

DIVE_REPORT = {
    "en": {
        "verdict": "active",
        "summary": "Choice complexity and price obfuscation are well studied, but almost always as drivers of purchase, not of later churn. "
                   "The specific link from tier naming to retention does not appear in this sample.",
        "known": [
            {"point": "Adding a dominated option raises the choice share of the option that dominates it — the classic decoy effect.", "refs": [1]},
            {"point": "Large assortments reduced purchase in a famous field experiment, but meta-analyses find the average overload effect is small and highly variable.", "refs": [2, 3]},
            {"point": "Overload depends on moderators such as choice-set complexity and preference uncertainty — plan naming plausibly raises both.", "refs": [4]},
            {"point": "Sellers deliberately make prices hard to compare, and shrouded attributes can survive competition.", "refs": [5, 6]},
            {"point": "Subscribers systematically mispredict their own usage and are slow to cancel flat-rate contracts.", "refs": [7]},
        ],
        "gap": "The sample links complexity to the purchase moment, and subscription behaviour to usage forecasts, but not complexity at purchase to churn afterwards.",
        "sharpened_question": "Do abstract tier names increase 90-day churn by causing mismatched plan choices, rather than by price level?",
        "approaches": [
            {"design": "Cross-firm comparison: code tier-name abstractness for panel firms; regress early churn on it with price-dispersion controls.", "data": "Your firm panel plus archived pricing pages (Wayback Machine)."},
            {"design": "Within-firm event study around a renaming of tiers.", "data": "Firms in the panel that renamed plans; dates from archived pages."},
        ],
        "next_steps": ["List panel firms that renamed tiers in the sample period", "Write a two-line coding rule for 'descriptive' vs 'abstract'", "Read the meta-analysis moderators section before designing the coding"],
        "risks": ["Abstract naming may proxy for product complexity", "Archived pricing pages are incomplete for small firms"],
        "novelty_note": "Eight works cannot establish novelty. A proper search of marketing journals for 'plan naming' and 'menu labels' in subscriptions is still needed.",
        "cited": [1, 2, 3, 4, 5, 6, 7],
    },
    "zh": {
        "verdict": "active",
        "summary": "选择复杂度与价格混淆研究得很多，但几乎都把它们当作购买的驱动因素，而不是事后流失的驱动因素。在这个样本里，没有出现从套餐命名到留存的直接联系。",
        "known": [
            {"point": "加入一个被占优的选项，会提高占优它的那个选项的份额——经典的诱饵效应。", "refs": [1]},
            {"point": "一个著名的田野实验发现大规模选项降低了购买，但元分析显示平均的选择过载效应很小且差异很大。", "refs": [2, 3]},
            {"point": "过载取决于选项集复杂度、偏好不确定性等调节变量——套餐命名很可能同时提高这两者。", "refs": [4]},
            {"point": "卖家会刻意让价格难以比较，而被遮蔽的属性可以在竞争中存活。", "refs": [5, 6]},
            {"point": "订阅者会系统性地错估自己的使用量，并且迟迟不取消包月合同。", "refs": [7]},
        ],
        "gap": "样本把复杂度联系到购买时刻，把订阅行为联系到使用预期，但没有把购买时的复杂度联系到之后的流失。",
        "sharpened_question": "抽象的套餐名称是否通过造成套餐错配（而非价格水平）提高了 90 天流失？",
        "approaches": [
            {"design": "跨公司比较：为面板公司的套餐名称抽象程度编码，在控制价格离散度后回归早期流失。", "data": "你的企业面板 + 存档的定价页面（Wayback Machine）。"},
            {"design": "公司内事件研究：围绕套餐改名前后。", "data": "面板中改过套餐名的公司；日期来自存档页面。"},
        ],
        "next_steps": ["列出样本期内改过套餐名的面板公司", "写两行“描述性/抽象”的编码规则", "设计编码前先读元分析的调节变量部分"],
        "risks": ["抽象命名可能是产品复杂度的代理变量", "小公司的存档定价页不完整"],
        "novelty_note": "八篇文献无法证明新颖性，仍需要在营销期刊中系统检索订阅情境下的“套餐命名”和“菜单标签”。",
        "cited": [1, 2, 3, 4, 5, 6, 7],
    },
}


def _evidence(app, day: str, tags: list[str], thread_ids: dict[str, int]) -> list[dict[str, Any]]:
    subjects = app.store.subjects(day)
    mapping = app.store.subject_threads(day)
    out = []
    for tag in tags:
        kind, _, key = tag.partition(":")
        tid = thread_ids.get(key)
        want = {"jot": "jot", "search": "search", "reading": "reading"}.get(kind, "focus")
        match = next((s for s in subjects if mapping.get(s["subject_key"]) == tid and s["kind"] == want), None)
        if match:
            out.append({
                "ref": "", "label": match["subject"], "kind": match["kind"], "seconds": round(match["seconds"] or 0),
                "visits": match["visits"], "url": match["url"] or "", "domain": match["domain"] or "",
                "body": (match.get("body") or "")[:400], "day": day,
            })
    for index, item in enumerate(out, 1):
        item["ref"] = f"S{index}"
    return out


def _write_dreams(app, today: date, thread_ids: dict[str, int], language: str, compute_signals) -> int:
    store = app.store
    count = 0
    for dream in sorted(DREAMS[language], key=lambda d: -d["offset"]):  # oldest first: dive numbers follow the days
        day = (today - timedelta(days=dream["offset"])).isoformat()
        signals, _ = compute_signals(app, day, language)
        digest = store.digest(day)
        topics = []
        threads = {t["id"]: t for t in store.threads()}
        for topic in (digest or {}).get("payload", {}).get("topics", []):
            thread = threads.get(topic.get("thread_id"))
            topics.append({**topic, "thread_name": thread["name"] if thread else None, "hue": thread["hue"] if thread else None})
        total = store.day_seconds(day)
        dream_id = store.save_dream(
            day,
            title=dream["title"],
            reflection=dream["reflection"],
            undercurrent=dream["undercurrent"],
            payload={
                "topics": topics,
                "signals": [s.to_dict() for s in signals],
                "stats": {"seconds": round(total), "subjects": len(store.subjects(day)), "threads": len(topics),
                          "candidates": 6, "kept": len(dream["sparks"])},
                "rejected": [],
                "demo": True,
            },
            models={"digest": "demo", "dream": "sample", "critique": "sample"},
        )
        when = datetime.fromisoformat(f"{day}T{app.settings.dream.time}:00").astimezone()
        with store.tx() as db:  # each borrowed night dived at its dive time, whenever the sea is borrowed
            db.execute("UPDATE dreams SET created_at=? WHERE id=?", (when.isoformat(timespec="seconds"), dream_id))
        count += 1
        for spark in dream["sparks"]:
            spark_id = store.add_spark(
                dream_id=dream_id, day=day, title=spark["title"], mechanism=spark["mechanism"], question=spark["question"],
                insight=spark["insight"], first_step=spark["first_step"], kill=spark["kill"], field=spark["field"],
                thread_ids=[thread_ids[k] for k in spark["threads"] if k in thread_ids],
                evidence=_evidence(app, day, spark["evidence"], thread_ids),
                search_terms=spark["search_terms"], scores=spark["scores"], score=float(spark["score"]),
                objection=spark["objection"], status=spark.get("status", "new"), reason=spark.get("reason", ""),
            )
            if spark.get("dive"):
                store.add_dive(spark_id, DIVE_REPORT[language], DIVE_PAPERS, "sample")
    return count
