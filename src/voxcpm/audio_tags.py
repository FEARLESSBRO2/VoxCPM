"""External audio-tag interpreter for VoxCPM.

Parses Gemini-style ``[tag]`` markup that VoxCPM cannot understand natively and
turns it into per-segment VoxCPM control instructions plus inserted silences.
Pure standard library at import time; numpy is imported lazily in the driver.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass

# --- family base adjectives (hybrid fallback) -------------------------------
FAMILY_ADJECTIVES: dict[str, str] = {
    "curiosity": "curious",
    "fear": "fearful",
    "sadness": "sad",
    "anger": "angry",
    "surprise": "surprised",
    "positive": "warm",
    "confidence": "confident",
}

# --- every emotion tag grouped by family (inverted into TAG_FAMILIES below) -
_FAMILY_TAGS: dict[str, list[str]] = {
    "curiosity": [
        "curiosity", "interest", "intrigue", "fascination", "anticipation",
        "suspicion", "skepticism", "doubt", "uncertainty", "unclear",
        "confusion", "ambivalence", "mixed", "concern", "caution", "cautious",
        "apprehension", "unease", "nervousness", "anxiety", "tension",
        "agitation", "speculation", "contemplative", "pensive", "reflection",
        "observation", "awareness", "awe", "amazement", "astonishment",
        "awkwardness", "discomfort",
    ],
    "fear": [
        "fear", "dread", "horror", "terror", "panic", "alarm", "distress",
        "desperation", "stress", "struggle", "suffering", "pain", "hurt",
        "urgency", "warning", "defiance",
    ],
    "sadness": [
        "sadness", "melancholy", "grief", "despair", "despondency",
        "disappointment", "disillusionment", "regret", "guilt", "shame",
        "embarrassment", "embitterment", "resignation", "pity", "sympathy",
        "compassion", "empathy", "nostalgia", "reminiscence", "sentimentality",
        "wistful", "yearning", "exhaustion", "tiredness", "weariness",
        "boredom", "worry", "pessimism", "indifference",
    ],
    "anger": [
        "anger", "aggression", "indignation", "annoyance", "frustration",
        "offense", "arrogance", "challenging", "critical", "criticism",
        "admonition", "disagreement", "disapproval", "dismissive", "contempt",
        "disdain", "disgust", "aversion", "dislike", "sarcasm",
        "self-deprecation",
    ],
    "surprise": [
        "surprise", "positive surprise", "negative surprise", "shock",
        "incredulity", "disbelief", "realization", "recognition",
        "understanding", "discernment",
    ],
    "positive": [
        "happy", "joy", "enjoyment", "amusement", "amused", "humor", "playful",
        "excitement", "enthusiasm", "enthusiastic", "eagerness", "thrill",
        "effervescence", "optimism", "hope", "relief", "slight relief",
        "comfort", "contentment", "satisfaction", "gratification",
        "self-satisfaction", "serenity", "relaxation", "pleased", "pride",
        "triumph", "victory", "success", "accomplishment", "achievement",
        "acceptance", "approval", "appreciation", "admiration", "adoration",
        "affection", "fondness", "love", "caring", "devotion", "gratitude",
        "thanks", "praise", "encouraging", "friendly", "enchantment",
        "smitten", "craving", "desire", "passion", "animation", "embracement",
        "positive",
    ],
    "confidence": [
        "confidence", "confident", "conviction", "certainty", "assurance",
        "courage", "determination", "determined", "assertion", "assertive",
        "assertiveness", "directness", "seriousness", "solemnity", "analysis",
        "logical reasoning", "explaining", "informative", "instruction",
        "description", "descriptive", "demonstration", "summary", "planning",
        "strategizing", "decision", "thinking", "focus", "concentration",
        "suggestion", "invitation", "bargaining", "apology", "pleading",
        "wisdom", "emphasis", "neutral", "negative",
    ],
}

TAG_FAMILIES: dict[str, str] = {
    tag: family for family, tags in _FAMILY_TAGS.items() for tag in tags
}

# --- per-tag overrides (hybrid): only where the family word loses nuance ----
TAG_ADJECTIVES: dict[str, str] = {
    "fear": "fearful, tense",
    "dread": "ominous, dreadful",
    "terror": "terrified",
    "horror": "horrified",
    "panic": "panicked",
    "suspicion": "suspicious",
    "tension": "tense",
    "unease": "uneasy",
    "nervousness": "nervous",
    "shock": "shocked",
    "disbelief": "disbelieving",
    "realization": "realizing",
    "grief": "grieving",
    "nostalgia": "wistful, warm",
    "seriousness": "serious, grave",
    "whispers": "whispering, breathy",
}

# --- mechanics tags ---------------------------------------------------------
PACE_TAGS: dict[str, str] = {"slow": "slow", "fast": "fast"}
ENERGY_TAGS: dict[str, str] = {
    "low energy": "low energy",
    "high energy": "high energy",
    "energetic": "energetic",
    "active": "active",
    "passive": "passive",
}
PAUSE_TAGS: dict[str, float] = {"short pause": 1.0, "long pause": 2.0}
SKIP_TAGS: set[str] = {"laughs"}


@dataclass
class Segment:
    """Audio segment with kind, text, control instructions, and optional duration."""
    kind: str            # "speech" | "silence"
    text: str = ""
    control: str = ""
    duration: float = 0.0


def build_control(emotion: str = "", pace: str = "", energy: str = "") -> str:
    """Comma-join the active adjective + pace + energy, skipping empties."""
    return ", ".join(p for p in (emotion, pace, energy) if p)


def _emotion_adjective(tag: str) -> str:
    """Resolve a known emotion/style tag to its control adjective.

    Per-tag override wins; otherwise the family base adjective is used.
    ``tag`` must be present in TAG_ADJECTIVES or TAG_FAMILIES.
    """
    if tag in TAG_ADJECTIVES:
        return TAG_ADJECTIVES[tag]
    return FAMILY_ADJECTIVES[TAG_FAMILIES[tag]]


_TAG_RE = re.compile(r"\[([^\]]+)\]")


def parse_tagged_script(
    text: str,
    *,
    short_pause: float = PAUSE_TAGS["short pause"],
    long_pause: float = PAUSE_TAGS["long pause"],
) -> list[Segment]:
    """Parse ``[tag]`` markup into an ordered list of speech/silence Segments.

    Emotion/pace/energy state persists across untagged text until changed.
    Pause tags emit silence and never alter emotion/pace/energy. Unknown tags
    are stripped and warned. Adjacent speech with identical control is merged.
    """
    pause_map = {"short pause": short_pause, "long pause": long_pause}
    cur_emotion = cur_pace = cur_energy = ""
    raw: list[Segment] = []

    # re.split with a capturing group yields: text, tag, text, tag, ... text
    parts = _TAG_RE.split(text)
    for i, part in enumerate(parts):
        if i % 2 == 0:  # plain text
            speech = part.strip()
            if speech:
                raw.append(Segment(
                    "speech",
                    text=speech,
                    control=build_control(cur_emotion, cur_pace, cur_energy),
                ))
            continue

        tag = part.strip().lower()
        if tag in pause_map:
            raw.append(Segment("silence", duration=pause_map[tag]))
        elif tag in PACE_TAGS:
            cur_pace = PACE_TAGS[tag]
        elif tag in ENERGY_TAGS:
            cur_energy = ENERGY_TAGS[tag]
        elif tag in SKIP_TAGS:
            pass
        elif tag in TAG_ADJECTIVES or tag in TAG_FAMILIES:
            cur_emotion = _emotion_adjective(tag)
        else:
            warnings.warn(f"Unknown audio tag [{tag}] — stripped, not spoken")

    # merge adjacent speech segments with identical control
    merged: list[Segment] = []
    for seg in raw:
        if (
            seg.kind == "speech"
            and merged
            and merged[-1].kind == "speech"
            and merged[-1].control == seg.control
        ):
            merged[-1].text += " " + seg.text
        else:
            merged.append(seg)
    return merged
