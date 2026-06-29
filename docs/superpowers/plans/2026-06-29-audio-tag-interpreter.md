# Audio Tag Interpreter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let VoxCPM render Gemini-style audio tags (`[fear]`, `[whispers]`, `[long pause]`) that the model itself does not understand, via an external parser + driver — no model change.

**Architecture:** A pure-Python parser turns a tagged script into an ordered list of speech/silence `Segment`s, mapping each emotion/pace/energy tag to a short VoxCPM control instruction (hybrid: per-tag override, else family fallback). A thin driver calls VoxCPM `generate` per speech segment, inserts zero-sample silence for pause segments, and concatenates. A one-method wrapper on `VoxCPM` wires the two together.

**Tech Stack:** Python 3.10+, stdlib only for the parser (`re`, `dataclasses`, `typing`, `warnings`), numpy (lazy-imported) for the driver. Tests: pytest, loading target modules via `importlib` (the `voxcpm` package is not pip-installed in CI; existing tests in `tests/` use this pattern).

## Global Constraints

- Parser file imports stdlib only — NO `torch`, NO `numpy` at module top. `numpy` is imported lazily inside the driver function only.
- Tags are case-insensitive, matched as whole bracketed tokens `[tag]`; tags are always English even though narration is Hindi.
- Unknown bracketed tokens are stripped, warned via `warnings.warn`, and NEVER spoken. This is the explicit fix for VoxCPM speaking tag words aloud.
- Pause durations: `[short pause]` = 1.0 s, `[long pause]` = 2.0 s (exposed as overridable params).
- `[laughs]` (and the `SKIP_TAGS` set) is stripped and produces no segment in v1.
- `[whispers]` is a style adjective `whispering, breathy`, not a sound effect.
- Control strings kept short (VoxCPM favors short control; long descriptors degrade output).
- Tests are GPU-free and do not require the model weights or a pip-installed `voxcpm`.
- Match existing repo test style: load single source files with `importlib.util.spec_from_file_location` rather than `import voxcpm` (see `tests/test_model_utils.py`, `tests/test_validate.py`).

---

## File Structure

- **Create `src/voxcpm/audio_tags.py`** — tag tables, `Segment` dataclass, `build_control`, `parse_tagged_script` (parser), `synthesize_tagged_script` (driver). One file: these pieces change together and share the tag tables.
- **Modify `src/voxcpm/core.py`** — add one method `generate_from_tagged_script` on the `VoxCPM` class (after `generate_streaming`, line 178). Wires `self.generate` + `self.tts_model.sample_rate` into the parser/driver.
- **Create `tests/test_audio_tags.py`** — parser unit tests, driver unit test (fake `generate`), core wiring test (stubbed heavy imports).

---

### Task 1: Tag tables, `Segment`, and `build_control`

**Files:**
- Create: `src/voxcpm/audio_tags.py`
- Test: `tests/test_audio_tags.py`

**Interfaces:**
- Consumes: nothing.
- Produces:
  - `FAMILY_ADJECTIVES: dict[str, str]` — 7 family keys → base adjective.
  - `TAG_FAMILIES: dict[str, str]` — every emotion tag → family key.
  - `TAG_ADJECTIVES: dict[str, str]` — per-tag adjective overrides (incl. `"whispers"`).
  - `PACE_TAGS: dict[str, str]`, `ENERGY_TAGS: dict[str, str]`, `PAUSE_TAGS: dict[str, float]`, `SKIP_TAGS: set[str]`.
  - `@dataclass Segment` — `kind: str`, `text: str = ""`, `control: str = ""`, `duration: float = 0.0`.
  - `build_control(emotion: str = "", pace: str = "", energy: str = "") -> str`.
  - `_emotion_adjective(tag: str) -> str` — resolves a known emotion tag to its adjective.

- [ ] **Step 1: Write the failing test**

Create `tests/test_audio_tags.py`:

```python
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AT_PATH = ROOT / "src" / "voxcpm" / "audio_tags.py"


def _load_audio_tags():
    spec = importlib.util.spec_from_file_location("voxcpm_audio_tags", AT_PATH)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


at = _load_audio_tags()


def test_tag_tables_cover_212_tags():
    emotion = set(at.TAG_FAMILIES)
    mechanics = set(at.PACE_TAGS) | set(at.ENERGY_TAGS) | set(at.PAUSE_TAGS) | at.SKIP_TAGS | {"whispers"}
    assert len(emotion) == 201
    assert len(mechanics) == 11
    assert len(emotion | mechanics) == 212


def test_build_control_joins_nonempty():
    assert at.build_control("fearful, tense", "slow", "low energy") == "fearful, tense, slow, low energy"
    assert at.build_control("", "fast", "") == "fast"
    assert at.build_control("", "", "") == ""


def test_emotion_adjective_override_beats_family():
    assert at._emotion_adjective("fear") == "fearful, tense"   # override
    assert at._emotion_adjective("interest") == "curious"      # family fallback (curiosity)
    assert at._emotion_adjective("whispers") == "whispering, breathy"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: FAIL — `FileNotFoundError` / `ModuleNotFoundError` (audio_tags.py does not exist yet).

- [ ] **Step 3: Write minimal implementation**

Create `src/voxcpm/audio_tags.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: PASS (3 tests).

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm/audio_tags.py tests/test_audio_tags.py
git commit -m "feat(audio_tags): tag tables, Segment, build_control"
```

---

### Task 2: `parse_tagged_script` (the parser)

**Files:**
- Modify: `src/voxcpm/audio_tags.py` (append the parser; add `re`/`warnings` already imported in Task 1)
- Test: `tests/test_audio_tags.py` (append parser tests)

**Interfaces:**
- Consumes (from Task 1): `Segment`, `build_control`, `_emotion_adjective`, `PACE_TAGS`, `ENERGY_TAGS`, `PAUSE_TAGS`, `SKIP_TAGS`, `TAG_ADJECTIVES`, `TAG_FAMILIES`.
- Produces: `parse_tagged_script(text: str, *, short_pause: float = PAUSE_TAGS["short pause"], long_pause: float = PAUSE_TAGS["long pause"]) -> list[Segment]`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_tags.py`:

```python
import warnings as _warnings


def _speech(segs):
    return [(s.text, s.control) for s in segs if s.kind == "speech"]


def test_single_emotion_tag():
    segs = at.parse_tagged_script("[fear]Hello")
    assert _speech(segs) == [("Hello", "fearful, tense")]


def test_emotion_persists_until_next_emotion_tag():
    segs = at.parse_tagged_script("[fear]One. Two.")
    assert _speech(segs) == [("One. Two.", "fearful, tense")]

    segs = at.parse_tagged_script("[fear]One [suspicion]Two")
    assert _speech(segs) == [("One", "fearful, tense"), ("Two", "suspicious")]


def test_pace_and_energy_compose_in_order():
    segs = at.parse_tagged_script("[fear][slow][low energy]Hi")
    assert _speech(segs) == [("Hi", "fearful, tense, slow, low energy")]


def test_pause_tags_emit_silence_segments():
    short = at.parse_tagged_script("[short pause]")
    assert len(short) == 1
    assert short[0].kind == "silence" and short[0].duration == 1.0

    long = at.parse_tagged_script("[long pause]")
    assert long[0].kind == "silence" and long[0].duration == 2.0

    custom = at.parse_tagged_script("[short pause]", short_pause=0.5)
    assert custom[0].duration == 0.5


def test_pause_does_not_reset_emotion():
    segs = at.parse_tagged_script("[fear]Hi[long pause]Bye")
    assert _speech(segs) == [("Hi", "fearful, tense"), ("Bye", "fearful, tense")]
    assert [s.kind for s in segs] == ["speech", "silence", "speech"]


def test_whispers_sets_style_adjective():
    segs = at.parse_tagged_script("[whispers]secret")
    assert _speech(segs) == [("secret", "whispering, breathy")]


def test_laughs_is_skipped_and_emits_nothing():
    segs = at.parse_tagged_script("[laughs]Hello")
    assert _speech(segs) == [("Hello", "")]


def test_unknown_tag_stripped_warns_and_never_spoken():
    with _warnings.catch_warnings(record=True) as caught:
        _warnings.simplefilter("always")
        segs = at.parse_tagged_script("[bogus]Hello")
    assert _speech(segs) == [("Hello", "")]
    assert any("bogus" in str(w.message) for w in caught)
    assert all("bogus" not in s.text for s in segs)


def test_adjacent_identical_control_merges():
    segs = at.parse_tagged_script("[fear]A[fear]B")
    assert _speech(segs) == [("A B", "fearful, tense")]


def test_no_tags_single_empty_control_segment():
    segs = at.parse_tagged_script("Hello world")
    assert _speech(segs) == [("Hello world", "")]


def test_whitespace_only_text_dropped():
    segs = at.parse_tagged_script("[fear]   [suspicion]Real")
    assert _speech(segs) == [("Real", "suspicious")]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_tags.py -k "parse or pause or whispers or laughs or unknown or adjacent or no_tags or emotion or pace or whitespace" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'parse_tagged_script'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/voxcpm/audio_tags.py`:

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: PASS (all Task 1 + Task 2 tests).

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm/audio_tags.py tests/test_audio_tags.py
git commit -m "feat(audio_tags): parse_tagged_script parser"
```

---

### Task 3: `synthesize_tagged_script` (the driver)

**Files:**
- Modify: `src/voxcpm/audio_tags.py` (append the driver)
- Test: `tests/test_audio_tags.py` (append driver tests)

**Interfaces:**
- Consumes (from Tasks 1–2): `Segment`, `parse_tagged_script`.
- Produces: `synthesize_tagged_script(generate, segments, *, sample_rate, silence_dtype=None, **generate_kwargs) -> "np.ndarray"`. `generate` is a callable invoked as `generate(text=..., **generate_kwargs)` returning a 1-D float array. Raises `ValueError` when `segments` is empty.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_tags.py`:

```python
import numpy as np
import pytest


def _fake_generate(records):
    def gen(text, **kwargs):
        records.append(text)
        return np.ones(4, dtype=np.float32)
    return gen


def test_speech_segment_uses_control_prefix():
    records = []
    segs = [at.Segment("speech", text="Hi", control="fearful")]
    out = at.synthesize_tagged_script(_fake_generate(records), segs, sample_rate=16000)
    assert records == ["(fearful)Hi"]
    assert out.shape[0] == 4


def test_speech_segment_without_control_has_no_prefix():
    records = []
    segs = [at.Segment("speech", text="Hi", control="")]
    at.synthesize_tagged_script(_fake_generate(records), segs, sample_rate=16000)
    assert records == ["Hi"]


def test_silence_segment_inserts_zero_samples():
    segs = [at.Segment("silence", duration=2.0)]
    out = at.synthesize_tagged_script(_fake_generate([]), segs, sample_rate=10)
    assert out.shape[0] == 20
    assert not out.any()  # all zeros


def test_output_is_concatenation_in_order():
    records = []
    segs = [
        at.Segment("speech", text="A", control="x"),
        at.Segment("silence", duration=1.0),
        at.Segment("speech", text="B", control="x"),
    ]
    out = at.synthesize_tagged_script(_fake_generate(records), segs, sample_rate=10)
    # 4 (speech) + 10 (silence) + 4 (speech) = 18
    assert out.shape[0] == 18
    assert records == ["(x)A", "(x)B"]


def test_empty_segments_raises():
    with pytest.raises(ValueError):
        at.synthesize_tagged_script(_fake_generate([]), [], sample_rate=16000)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_tags.py -k "speech_segment or silence_segment or concatenation or empty_segments" -v`
Expected: FAIL — `AttributeError: module ... has no attribute 'synthesize_tagged_script'`.

- [ ] **Step 3: Write minimal implementation**

Append to `src/voxcpm/audio_tags.py`:

```python
def synthesize_tagged_script(
    generate,
    segments,
    *,
    sample_rate,
    silence_dtype=None,
    **generate_kwargs,
):
    """Drive VoxCPM over parsed segments and concatenate into one waveform.

    ``generate`` is a callable (e.g. ``VoxCPM.generate``) invoked per speech
    segment as ``generate(text=..., **generate_kwargs)``. Silence segments
    become ``int(duration * sample_rate)`` zero samples. numpy is imported
    here so the parser stays numpy-free.
    """
    import numpy as np

    if not segments:
        raise ValueError("no segments to synthesize (empty or whitespace-only input)")
    if silence_dtype is None:
        silence_dtype = np.float32

    chunks = []
    for seg in segments:
        if seg.kind == "speech":
            text = f"({seg.control}){seg.text}" if seg.control else seg.text
            chunks.append(generate(text=text, **generate_kwargs))
        else:  # silence
            n = int(seg.duration * sample_rate)
            chunks.append(np.zeros(n, dtype=silence_dtype))
    return np.concatenate(chunks)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: PASS (all tests so far).

- [ ] **Step 5: Commit**

```bash
git add src/voxcpm/audio_tags.py tests/test_audio_tags.py
git commit -m "feat(audio_tags): synthesize_tagged_script driver"
```

---

### Task 4: `VoxCPM.generate_from_tagged_script` wiring

**Files:**
- Modify: `src/voxcpm/core.py` (add method after `generate_streaming`, currently line 178)
- Test: `tests/test_audio_tags.py` (append core wiring test)

**Interfaces:**
- Consumes (from Tasks 2–3): `parse_tagged_script`, `synthesize_tagged_script` (relative import `from .audio_tags import ...`).
- Produces: `VoxCPM.generate_from_tagged_script(self, text, *, short_pause=1.0, long_pause=2.0, **generate_kwargs) -> np.ndarray`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_audio_tags.py`:

```python
import sys
import types


def _load_core():
    """Load src/voxcpm/core.py with heavy deps stubbed (no torch/model load)."""
    pkg = types.ModuleType("voxcpm")
    pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
    sys.modules["voxcpm"] = pkg

    hf = types.ModuleType("huggingface_hub")
    hf.snapshot_download = lambda *a, **k: None
    sys.modules["huggingface_hub"] = hf

    model_pkg = types.ModuleType("voxcpm.model")
    model_pkg.__path__ = []
    mv = types.ModuleType("voxcpm.model.voxcpm")
    mv.VoxCPMModel = type("VoxCPMModel", (), {})
    mv.LoRAConfig = type("LoRAConfig", (), {})
    mv2 = types.ModuleType("voxcpm.model.voxcpm2")
    mv2.VoxCPM2Model = type("VoxCPM2Model", (), {})
    mu = types.ModuleType("voxcpm.model.utils")
    mu.next_and_close = lambda x: x
    sys.modules.update({
        "voxcpm.model": model_pkg,
        "voxcpm.model.voxcpm": mv,
        "voxcpm.model.voxcpm2": mv2,
        "voxcpm.model.utils": mu,
    })

    path = ROOT / "src" / "voxcpm" / "core.py"
    spec = importlib.util.spec_from_file_location("voxcpm.core", path)
    core = importlib.util.module_from_spec(spec)
    sys.modules["voxcpm.core"] = core
    spec.loader.exec_module(core)
    return core


def test_generate_from_tagged_script_wires_parser_and_driver():
    core = _load_core()
    vox = core.VoxCPM.__new__(core.VoxCPM)  # bypass __init__ (no model load)
    vox.tts_model = types.SimpleNamespace(sample_rate=10)

    calls = []

    def fake_gen(text, **kwargs):
        calls.append(text)
        return np.ones(2, dtype=np.float32)

    vox.generate = fake_gen  # instance attr shadows the class method

    out = vox.generate_from_tagged_script("[fear]Hi[long pause]Bye")
    # speech Hi (2) + silence 2.0*10 (20) + speech Bye (2) = 24
    assert out.shape[0] == 24
    assert calls == ["(fearful, tense)Hi", "(fearful, tense)Bye"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_audio_tags.py -k "generate_from_tagged_script" -v`
Expected: FAIL — `AttributeError: 'VoxCPM' object has no attribute 'generate_from_tagged_script'`.

- [ ] **Step 3: Write minimal implementation**

In `src/voxcpm/core.py`, insert this method immediately after `generate_streaming` (after line 178, before `_generate`):

```python
    def generate_from_tagged_script(
        self,
        text: str,
        *,
        short_pause: float = 1.0,
        long_pause: float = 2.0,
        **generate_kwargs,
    ) -> np.ndarray:
        """Synthesize a script containing Gemini-style ``[tag]`` audio tags.

        Parses the tags into speech/silence segments (emotion/pace/energy →
        VoxCPM control instructions, pause tags → inserted silence), then drives
        ``self.generate`` per speech segment and concatenates. All other kwargs
        (``reference_wav_path``, ``prompt_wav_path``, ``prompt_text``,
        ``cfg_value``, ``normalize``, etc.) pass straight through unchanged so
        timbre/clone settings stay stable across segments.
        """
        from .audio_tags import parse_tagged_script, synthesize_tagged_script

        segments = parse_tagged_script(
            text, short_pause=short_pause, long_pause=long_pause
        )
        return synthesize_tagged_script(
            self.generate,
            segments,
            sample_rate=self.tts_model.sample_rate,
            **generate_kwargs,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: PASS (all parser, driver, and wiring tests).

- [ ] **Step 5: Run the full test file once more and commit**

Run: `python -m pytest tests/test_audio_tags.py -v`
Expected: PASS (no regressions).

```bash
git add src/voxcpm/core.py tests/test_audio_tags.py
git commit -m "feat(core): add VoxCPM.generate_from_tagged_script"
```

---

## Self-Review

**1. Spec coverage:**
- Tag vocabulary (212 tags, 7 emotion families + mechanics) → Task 1 (`_FAMILY_TAGS`, mechanics dicts; `test_tag_tables_cover_212_tags`). ✓
- Emotion/pace/energy mapping, hybrid override → Task 1 (`_emotion_adjective`, `build_control`). ✓
- Parser: state persistence, pause silence, whispers style, laughs skip, unknown strip+warn, merge, whitespace drop, no-tags → Task 2. ✓
- Driver: control prefix, silence zero samples, concatenation order, empty→ValueError → Task 3. ✓
- Core wiring + kwargs passthrough + sample_rate → Task 4. ✓
- Pauses 1.0 s / 2.0 s → `PAUSE_TAGS`, parser defaults, core defaults, tests. ✓
- "Tags but no speakable text (only `[long pause]`)" → silence-only waveform: covered by `test_pause_tags_emit_silence_segments` (parse) + driver concatenation of silence; totally empty raises `ValueError` (`test_empty_segments_raises`). ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; commands have expected output. ✓

**3. Type consistency:** `Segment(kind, text, control, duration)` used identically across Tasks 2–4; `parse_tagged_script` / `synthesize_tagged_script` / `generate_from_tagged_script` signatures match their consumers; `_emotion_adjective` defined in Task 1 and used in Task 2; `generate(text=..., **kwargs)` calling convention consistent between driver and core. ✓

## Limitations (carried from spec, not bugs)
- Emotion fidelity is best-effort — VoxCPM control is coarse; `[fear]`/`[dread]`/`[terror]` may sound similar.
- `[laughs]` and other non-verbal sounds are stripped, not synthesized.
- Per-segment generation is slower than one-shot (one `generate` call per control change).
