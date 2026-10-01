"""A deterministic stand-in model for tests and offline demos (DUN_FAKE_LLM=1).

It reads the structured ``payload`` of a request and answers with plausible,
schema-valid JSON built from simple heuristics. It is not meant to be clever;
it exists so every code path can run without an account or a network.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from unconscious.llm.base import LLMRequest, LLMResult
from unconscious.text import STOPWORDS, clip, jaccard, tokens

if TYPE_CHECKING:
    from unconscious.config import Settings
    from unconscious.store import Store

NOISE = {"chat", "system", "media", "social"}


@dataclass
class FakeLLM:
    name: str = "fake"
    default_model: str = "offline"

    def available(self) -> bool:
        return True

    def complete(self, request: LLMRequest, model: str | None) -> LLMResult:
        handler = getattr(self, f"_{request.role}", None)
        data = handler(request.payload) if handler else {"ok": True}
        return LLMResult(True, json.dumps(data, ensure_ascii=False), data, self.name, model or self.default_model, 5)

    # -- digest --------------------------------------------------------------

    def _digest(self, payload: dict[str, Any]) -> dict[str, Any]:
        subjects = [s for s in payload.get("subjects", []) if s.get("category") not in NOISE]
        threads = payload.get("threads", [])
        thread_tokens = {t["ref"]: tokens(t["name"] + " " + " ".join(t.get("keywords") or [])) for t in threads}
        groups: dict[str, list[dict[str, Any]]] = {}
        loose: list[tuple[dict[str, Any], set[str]]] = []
        for subject in subjects:
            toks = tokens(subject["label"] + " " + subject.get("body", ""))
            best = max(thread_tokens, key=lambda ref: len(toks & thread_tokens[ref]), default=None)
            strong = {w for w in toks & thread_tokens.get(best, set()) if len(w) >= 4} if best else set()
            if best and (len(toks & thread_tokens[best]) >= 2 or strong):
                groups.setdefault(best, []).append(subject)
            else:
                loose.append((subject, toks))
        clusters: list[tuple[set[str], list[dict[str, Any]]]] = []
        for subject, toks in loose:
            for cluster_tokens, members in clusters:
                if jaccard(toks, cluster_tokens) >= 0.12 or len(toks & cluster_tokens) >= 2:
                    members.append(subject)
                    cluster_tokens |= toks
                    break
            else:
                clusters.append((set(toks), [subject]))
        topics = []
        for ref, members in groups.items():
            name = next(t["name"] for t in threads if t["ref"] == ref)
            topics.append(self._topic(members, ref, name, ""))
        for cluster_tokens, members in clusters:
            if not cluster_tokens:
                continue
            counts = Counter(w for m in members for w in tokens(m["label"] + " " + m.get("body", "")) if w not in STOPWORDS)
            words = [w for w, _ in counts.most_common(3)]
            name = " ".join(words).title() if words else clip(members[0]["label"], 40)
            topics.append(self._topic(members, "new", name, f"Recurring interest in {', '.join(words)}."))
        return {"topics": topics}

    def _topic(self, members: list[dict[str, Any]], thread: str, name: str, gist: str) -> dict[str, Any]:
        counts = Counter(w for m in members for w in tokens(m["label"]))
        return {
            "label": name,
            "gist": f"Spent time on {clip(members[0]['label'], 80)}" + (f" and {len(members) - 1} related things." if len(members) > 1 else "."),
            "subjects": [m["ref"] for m in members],
            "thread": thread,
            "thread_name": name if thread == "new" else "",
            "thread_gist": gist if thread == "new" else "",
            "keywords": [w for w, _ in counts.most_common(5)],
        }

    # -- dream ---------------------------------------------------------------

    def _dream(self, payload: dict[str, Any]) -> dict[str, Any]:
        threads: dict[str, str] = payload.get("threads", {})
        topics = payload.get("topics", [])
        subjects = {s["subject_key"]: s for s in payload.get("subjects", [])}

        def evidence_for(thread_ids: list[int]) -> list[str]:
            refs = []
            for topic in topics:
                if topic.get("thread_id") in thread_ids:
                    refs += [subjects[k]["ref"] for k in topic["subject_keys"] if k in subjects]
            return refs[:3]

        sparks = []
        for signal in payload.get("signals", []):
            kind = signal["kind"]
            if kind in {"steady", "fade"}:
                continue
            refs = [f"T{tid}" for tid in signal["threads"] if f"T{tid}" in threads]
            if not refs:
                continue
            names = [threads[r] for r in refs]
            if kind == "collision" and len(names) == 2:
                title = f"What {names[0]} could borrow from {names[1]}"
                question = f"Does a mechanism studied in {names[1]} predict outcomes in {names[0]}?"
            else:
                title = f"The question under {names[0]}"
                question = f"What specific claim about {names[0]} would you bet on, and how would you check it?"
            sparks.append(self._spark(kind if kind != "surge" else "surge", title, question, refs,
                                      evidence_for(signal["threads"])))
        for topic in topics:
            if len(sparks) >= payload.get("candidates", 3):
                break
            ref = f"T{topic['thread_id']}" if topic.get("thread_id") else None
            ev = [subjects[k]["ref"] for k in topic["subject_keys"] if k in subjects][:3]
            if not ev:
                continue
            sparks.append(self._spark(
                "gap", f"A testable version of “{clip(topic['label'], 50)}”",
                f"What would a one-week measurement of {topic['label'].lower()} show?",
                [ref] if ref and ref in threads else [], ev,
            ))
        if not sparks and subjects:
            first = next(iter(subjects.values()))
            sparks.append(self._spark("seed", f"Where “{clip(first['subject'], 40)}” leads", "What is the sharpest question here?", [], [first["ref"]]))
        top = topics[0]["label"] if topics else "the day"
        return {
            "title": f"Circling {clip(top, 50).lower()}",
            "reflection": f"Most of your attention went to {top.lower()}. You kept returning to it between other things, "
                          "which usually means a question is still open.",
            "undercurrent": f"What would change if {top.lower()} turned out to be simpler than it looks?",
            "sparks": sparks[: max(1, payload.get("candidates", 3))],
        }

    def _spark(self, mechanism: str, title: str, question: str, threads: list[str], evidence: list[str]) -> dict[str, Any]:
        return {
            "title": title,
            "mechanism": mechanism,
            "threads": threads,
            "evidence": evidence,
            "question": question,
            "insight": "The evidence shows the same concern appearing in different contexts, which suggests it is "
                       "more than a passing interest.",
            "first_step": "Write down the claim in one sentence and find one public dataset that could test it.",
            "kill": "If the dataset cannot separate the two explanations, drop it.",
            "field": "",
            "search_terms": [title.lower()[:60]],
        }

    # -- critique ------------------------------------------------------------

    def _critique(self, payload: dict[str, Any]) -> dict[str, Any]:
        reviews = []
        for index, card in enumerate(payload.get("cards", []), 1):
            grounded = min(5, 2 + len(card.get("evidence_refs", [])))
            fresh = 4 if card.get("mechanism") in {"collision", "orbit", "return"} else 3
            reviews.append({
                "card": index, "grounded": grounded, "sharp": 3, "fresh": fresh, "doable": 4, "fit": 3,
                "generic": False, "objection": "The pattern may reflect what was convenient to read, not what matters.",
            })
        return {"reviews": reviews}

    # -- dive ----------------------------------------------------------------

    def _dive(self, payload: dict[str, Any]) -> dict[str, Any]:
        papers = payload.get("papers", [])
        known = [{"point": clip(p.get("abstract") or p["title"], 200), "refs": [i]} for i, p in enumerate(papers[:3], 1)]
        return {
            "verdict": "active" if len(papers) >= 5 else "unclear",
            "summary": f"The retrieved sample has {len(papers)} works; several touch the idea from adjacent angles.",
            "known": known,
            "gap": "None of the sampled works combines the two contexts directly.",
            "sharpened_question": payload.get("spark", {}).get("question", ""),
            "approaches": [{"design": "Observational comparison across matched groups", "data": "A public panel dataset"}],
            "next_steps": ["Read the two most cited works", "Sketch the comparison table"],
            "risks": ["Selection effects in observational data"],
            "novelty_note": "A sample of a dozen works cannot establish novelty; a systematic search is needed.",
        }

    def _ping(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {"ok": True}


def fake_router(settings: Settings, store: Store | None = None):
    import copy

    from unconscious.llm.router import Router

    # A private copy: the person's real settings must never be saved with "fake" in them.
    local = copy.deepcopy(settings)
    for role in ("digest", "dream", "critique", "dive"):
        setattr(local.models, role, "fake")
    local.models.fallback = False
    return Router(local, store, providers={"fake": FakeLLM()})
