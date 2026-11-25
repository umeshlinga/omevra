"""The human Copilot layer (Step 4, Items 21 + 23).

The gated Copilot answers like an auditor: correct, complete, and
about as warm as a lab report.  Real users — a student at midnight,
a professor between lectures, a researcher in a hurry — need a
guide, not a printout.  This module is that guide.

It sits strictly ON TOP of the gate:

* Every factual sentence about biology comes from a finished,
  gated :class:`~omevra.copilot.CopilotAnswer`.  The guide may
  choose order, depth, and tone; it may never add a claim, soften
  a decline, or hide a reason.
* A DECLINE stays a decline.  The guide's job there is kindness
  plus precision: say plainly what is missing and ask for exactly
  the thing that would unblock the answer.
* Questions about the tool itself ("what is Provome?", "how do you
  work?", "what does it cost?") are answered from a small static
  fact sheet about the product — the one topic the Copilot is
  allowed to describe without the evidence gate, because the facts
  are the product's own documentation, stated with its limits.
* Run transparency (Item 23): every guided answer carries a "how I
  got this" section — the topic recognised, the provider used, the
  gate arithmetic — so a user can see the work, not just the result.

Depth levels adapt the same gated answer: ``simple`` (plain words,
the one thing that matters), ``standard`` (the working answer), and
``expert`` (full claims, citations, and gate detail).

Running this module directly demonstrates all three behaviours
(offline provider; no network needed):

    python -m omevra.converse

Research software.  Not for clinical use.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from .copilot import Copilot, CopilotAnswer, Verdict
from .evidence import MIN_INDEPENDENT_SOURCES, MIN_VERIFIED_ITEMS


class Depth(str, Enum):
    SIMPLE = "simple"
    STANDARD = "standard"
    EXPERT = "expert"


TOOL_FACTS = {
    "what": (
        "Provome is an evidence engine for biology. You bring a "
        "problem — today, a gene and a variant — and Provome brings "
        "back an answer built only from checked evidence, with the "
        "sources named, or it honestly declines and tells you "
        "exactly what is missing. That rule is called cite-or-decline."
    ),
    "how": (
        "One Copilot speaks to you. Behind it, specialist workers "
        "gather evidence from registered sources, cross-check each "
        "other, and pass the result through quality stages and a "
        "senior review. An answer is issued only when at least "
        f"{MIN_VERIFIED_ITEMS} verified evidence items from at least "
        f"{MIN_INDEPENDENT_SOURCES} independent sources support it. "
        "Every run leaves an audit trail."
    ),
    "coverage": (
        "Honest coverage, today: the Human division is live and "
        "benchmarked for variant evidence (AUROC 0.9705 on the "
        "public benchmark). The Animal and Plant divisions and the "
        "computational Drug Discovery division are on the roadmap "
        "and are NOT claimed as covered until each is built and "
        "benchmarked. Provome is research software, not for "
        "clinical use."
    ),
    "price": (
        "The planned shape is freemium: a free tier with core tools "
        "open, a Researcher tier at $79, and a Lab tier at $499 "
        "unlocking everything. Checkout is not live yet; pilot "
        "access comes first."
    ),
    "sources": (
        "The live Copilot fetches from ClinVar (curated variant "
        "classifications), gnomAD (population frequencies, when "
        "coordinates are available), and peer-reviewed literature "
        "via Europe PMC / PubMed. Every citation carries the record "
        "identifier and the date it was fetched, so you can check "
        "it yourself."
    ),
}

_TOOL_PATTERNS = [
    (re.compile(r"what (is|are) (provome|omevra|this tool|you)\b"), "what"),
    (re.compile(r"how do (you|provome|omevra) work"), "how"),
    (re.compile(r"what can (you|provome|omevra) do"), "what"),
    (re.compile(r"(animal|plant) division|do you cover|your coverage|drug discovery division"), "coverage"),
    (re.compile(r"(price|pricing|how much|subscription|is it free|free tier)"), "price"),
    (re.compile(r"(what sources|which databases|where do you get|your sources)"), "sources"),
]


def tool_question_key(question_text: str) -> Optional[str]:
    """Return the TOOL_FACTS key a question asks about, or None."""
    lowered = question_text.lower()
    for pattern, key in _TOOL_PATTERNS:
        if pattern.search(lowered):
            return key
    return None


@dataclass
class GuidedAnswer:
    """A gated answer, guided for a human at a chosen depth."""

    kind: str  # "answer" | "decline" | "tool"
    opening: str
    body: str
    how_i_got_this: str = ""
    limits: str = ""
    follow_up: str = ""
    answer: Optional[CopilotAnswer] = None

    def render(self) -> str:
        parts = [self.opening, "", self.body]
        if self.how_i_got_this:
            parts += ["", "How I got this:", self.how_i_got_this]
        if self.limits:
            parts += ["", "Limits:", self.limits]
        if self.follow_up:
            parts += ["", self.follow_up]
        return "\n".join(parts)


_CLASSIFICATION_RE = re.compile(
    r"(?:classification:|classified as)\s*([A-Za-z][A-Za-z /]*?)(?:\s*\(|\.|,| for |$)"
)


class Guide:
    """Conversational layer over a Copilot.  Adds no claims, ever."""

    def __init__(self, copilot: Optional[Copilot] = None) -> None:
        self.copilot = copilot if copilot is not None else Copilot()

    def ask(self, question: str, depth: Depth = Depth.STANDARD) -> GuidedAnswer:
        if not isinstance(question, str):
            raise TypeError("question must be a string")
        key = tool_question_key(question)
        if key is not None:
            return GuidedAnswer(
                kind="tool",
                opening="Happy to explain — this part is about me, not the evidence gate.",
                body=TOOL_FACTS[key],
                follow_up="Ask me about a variant any time — for example: What is known about CFTR p.Phe508del?",
            )
        answer = self.copilot.ask(question)
        if answer.verdict is Verdict.ANSWER:
            return self._guide_answer(answer, depth)
        return self._guide_decline(answer, depth)

    # -- answers -------------------------------------------------------------
    def _guide_answer(self, answer: CopilotAnswer, depth: Depth) -> GuidedAnswer:
        topic = answer.topic_key or "your question"
        classification = None
        for claim in answer.claims:
            match = _CLASSIFICATION_RE.search(claim.statement)
            if match:
                classification = match.group(1).strip()
                break
        if depth is Depth.SIMPLE:
            opening = f"Good question — here's the short version on {topic}."
            plain = (
                f"In plain words: the checked evidence classifies this "
                f"variant as {classification}. "
                if classification
                else "In plain words: the checked evidence supports an answer. "
            )
            body = plain + (
                f"That rests on {answer.verified_item_count} verified "
                f"evidence items from {answer.independent_source_count} "
                f"independent sources — every one of them named below "
                f"in the full answer, so you can check them yourself."
            )
            if answer.claims:
                body += f"\n\nThe key record: {answer.claims[0].statement}"
                if answer.claims[0].citations:
                    body += f"\nKey source: {answer.claims[0].citations[0]}"
        else:
            opening = f"Here's what the evidence says about {topic}."
            lines = []
            for claim in answer.claims:
                lines.append(f"- {claim.statement}")
                if depth is Depth.EXPERT:
                    for citation in claim.citations:
                        lines.append(f"  Source: {citation}")
            body = "\n".join(lines)
            if depth is Depth.STANDARD and answer.claims and answer.claims[0].citations:
                body += f"\n\nKey source: {answer.claims[0].citations[0]}"
        how = (
            f"I recognised the topic {topic}, gathered evidence through "
            f"the {self._provider_name()} provider, and applied the "
            f"cite-or-decline gate: {answer.verified_item_count} verified "
            f"items (needs {MIN_VERIFIED_ITEMS}), "
            f"{answer.independent_source_count} independent source groups "
            f"(needs {MIN_INDEPENDENT_SOURCES}). The gate passed, so I "
            f"can answer. The confidence number "
            f"({answer.confidence:.2f}) is a heuristic about evidence "
            f"depth, not a probability that the answer is right."
        )
        limits = (
            "Research software — not for clinical use, and not a "
            "diagnosis. Classifications can change as new evidence "
            "arrives; the fetch dates on the citations tell you how "
            "current this run is."
        )
        follow = (
            "Want to go deeper? Ask me for the population frequency, "
            "the newest literature, or how this variant's mechanism "
            "works — I'll gather fresh evidence for each."
        )
        return GuidedAnswer(
            kind="answer", opening=opening, body=body,
            how_i_got_this=how, limits=limits, follow_up=follow,
            answer=answer,
        )

    # -- declines --------------------------------------------------------------
    def _guide_decline(self, answer: CopilotAnswer, depth: Depth) -> GuidedAnswer:
        opening = "I can't answer that yet — and I won't guess. Here's exactly why."
        if answer.topic_key is None:
            body = (
                "I couldn't recognise a specific gene and variant in "
                "your question, so I had nothing solid to gather "
                "evidence about. I decline rather than pattern-match "
                "my way to a confident-sounding answer."
            )
            follow = (
                "Try me in this shape: \"What is known about CFTR "
                "p.Phe508del?\" — a gene name plus the variant, and "
                "I'll fetch live evidence for it."
            )
        else:
            reasons = "; ".join(answer.declined_reasons) or "the evidence gate was not met"
            body = (
                f"I recognised the topic ({answer.topic_key}) and "
                f"gathered what I could, but the gate stopped me: "
                f"{reasons}. An answer from me has to rest on at "
                f"least {MIN_VERIFIED_ITEMS} verified items from "
                f"{MIN_INDEPENDENT_SOURCES} independent sources."
            )
            follow = (
                "If you have a paper, a lab report, or another angle "
                "on this variant, point me at it — or ask again later "
                "and I'll fetch fresh evidence."
            )
        how = (
            f"Provider: {self._provider_name()}. Gate arithmetic: "
            f"{answer.verified_item_count} verified items, "
            f"{answer.independent_source_count} independent source "
            f"groups — below the bar, so the only honest output is "
            f"this decline."
        )
        return GuidedAnswer(
            kind="decline", opening=opening, body=body,
            how_i_got_this=how, follow_up=follow, answer=answer,
        )

    def _provider_name(self) -> str:
        provider = getattr(self.copilot, "_provider", None) or getattr(
            self.copilot, "provider", None
        )
        return getattr(provider, "name", "offline-demo")


def _demo() -> None:
    guide = Guide()
    for question, depth in (
        ("What is Provome?", Depth.STANDARD),
        ("What is known about CFTR p.Phe508del?", Depth.SIMPLE),
        ("What is known about CFTR p.Phe508del?", Depth.EXPERT),
        ("Tell me about the weather in Indianapolis today please", Depth.STANDARD),
    ):
        print("=" * 72)
        print(f"Q ({depth.value}):", question)
        print(guide.ask(question, depth).render())
        print()


if __name__ == "__main__":
    _demo()
