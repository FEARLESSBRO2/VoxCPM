# Audio Tag Interpreter — Design Spec

> Date: 2026-06-29
> Status: approved for planning

## Goal

Let VoxCPM render Gemini-style audio tags (e.g. `[fear]`, `[whispers]`, `[long pause]`)
that VoxCPM itself does NOT understand. Confirmed by testing: VoxCPM speaks bracket
tags literally and supports no SSML/inline pause control (see
[OpenBMB/VoxCPM Issue #276](https://github.com/OpenBMB/VoxCPM/issues/276)). The fix is an
**external interpreter** that runs BEFORE generation — not a model change.

The interpreter is the **execution** half of a two-part system. The **insertion** half
already exists: `F:\story-pipeline\prompts\step02_scripting.txt` (the scripting LLM embeds
the tags). This spec covers execution only.

## Tag vocabulary (input)

The full tag set is the 212 tags documented in the step02 AUDIO TAG PERFORMANCE SYSTEM,
grouped into 8 families plus the mechanics group. Categories the interpreter recognizes:

- **Emotion tags** (~199): map to a control adjective phrase.
- **Pace tags**: `[slow]`, `[fast]`.
- **Energy tags**: `[low energy]`, `[high energy]`, `[energetic]`, `[active]`, `[passive]`.
- **Pause tags**: `[short pause]`, `[long pause]`.
- **Non-verbal sound tags**: `[whispers]`, `[laughs]`.
  - `[whispers]` is treated as a style adjective (`whispering, breathy`).
  - `[laughs]` is v1-skipped (see Limitations).

Tags are case-insensitive and matched as whole bracketed tokens `[tag]`.

## Architecture

```
tagged script
   │
   ▼
[ PARSER ]   pure Python, stdlib-only, GPU-free, unit-testable
   │  → ordered segment list (speech segments + silence segments)
   ▼
[ DRIVER ]   calls VoxCPM generate per speech segment, inserts silence, concatenates
   │
   ▼
final waveform (np.ndarray) @ model sample_rate
```

The split exists so the risky parsing logic is fully testable without a GPU. The driver is
thin and accepts the model's `generate` as an injected callable, so it is testable with a
fake.

## Components

### `src/voxcpm/audio_tags.py` (new — pure Python)

Holds the tag tables, the parser, and the control-string builder. Imports only the Python
standard library (`re`, `dataclasses`, `typing`). No `torch`, no `numpy`.

Public interface:

- `TAG_ADJECTIVES: dict[str, str]` — per-tag adjective overrides (e.g. `"fear" -> "fearful, tense"`,
  `"nostalgia" -> "wistful, warm"`, `"whispers" -> "whispering, breathy"`).
- `TAG_FAMILIES: dict[str, str]` — every emotion tag → one of 8 family keys; family supplies a
  fallback adjective when a tag has no explicit override.
- `PACE_TAGS: dict[str, str]` — `{"slow": "slow", "fast": "fast"}`.
- `ENERGY_TAGS: dict[str, str]` — `{"low energy": "low energy", "high energy": "high energy",
  "energetic": "energetic", "active": "active", "passive": "passive"}`.
- `PAUSE_TAGS: dict[str, float]` — `{"short pause": 0.30, "long pause": 0.70}`.
- `SKIP_TAGS: set[str]` — non-verbal sounds skipped in v1 (`{"laughs"}`).
- `@dataclass Segment` — `kind: Literal["speech","silence"]`, `text: str = ""`,
  `control: str = ""`, `duration: float = 0.0`.
- `parse_tagged_script(text: str, *, short_pause: float = 0.30, long_pause: float = 0.70)
  -> list[Segment]` — the parser.
- `build_control(emotion: str, pace: str, energy: str) -> str` — comma-joins the active
  adjective + pace word + energy word, skipping empties; returns `""` when nothing active.

### `src/voxcpm/audio_tags.py` driver helper

- `synthesize_tagged_script(generate, segments, *, sample_rate, silence_dtype=np.float32,
  **generate_kwargs) -> np.ndarray` — iterates segments; for `speech` calls
  `generate(text=f"({control}){seg.text}" if control else seg.text, **generate_kwargs)`; for
  `silence` makes `np.zeros(int(duration*sample_rate))`; concatenates in order. `numpy` is
  imported lazily inside this function so the parser stays numpy-free.
  `generate` is a callable (the caller passes `model.generate`), making this unit-testable
  with a fake that returns deterministic arrays.

### `src/voxcpm/core.py` (modified)

Add a thin method on `VoxCPM`:

- `generate_from_tagged_script(self, text, *, short_pause=0.30, long_pause=0.70,
  **generate_kwargs) -> np.ndarray` — parses then drives, wiring `self.generate` as the
  callable and `self.tts_model.sample_rate` as the sample rate. All other generate kwargs
  (`reference_wav_path`, `prompt_wav_path`, `prompt_text`, `cfg_value`, `normalize`, etc.)
  pass straight through unchanged so timbre/clone settings are stable across segments.

## Parser behavior (data flow detail)

Walk the text left to right, splitting on `[tag]` tokens. Maintain three pieces of state:
`current_emotion` (adjective string, default `""`), `current_pace` (default `""`),
`current_energy` (default `""`).

- Plain text between tags becomes a **speech** segment carrying
  `build_control(current_emotion, current_pace, current_energy)`.
- An **emotion tag** updates `current_emotion` to its adjective (override, else family
  fallback). Emotion state persists until the next emotion tag — matching the step02 rule
  "an untagged line keeps the mood of the previous tag".
- A **pace / energy tag** updates the corresponding state.
- A **pause tag** emits a **silence** segment of the mapped duration. It does not alter
  emotion/pace/energy state.
- A **`[whispers]`** tag sets the emotion adjective to `whispering, breathy` (so it composes
  with pace/energy).
- A **skip tag** (`[laughs]`) is removed and produces nothing.
- An **unknown** bracketed token is removed, logged via `warnings.warn`, and never spoken.
- Adjacent speech segments with identical control are merged into one (avoids redundant
  generate calls).
- Whitespace-only speech text is dropped.

## Error & edge handling

- **No tags at all** → one speech segment, empty control → identical to a plain
  `generate(text)` call (backward compatible).
- **Tags but no speakable text** (e.g. only `[long pause]`) → returns the silence-only
  waveform; if totally empty, raises `ValueError`.
- **Unknown tag** → stripped + warned, never pronounced. This is the explicit fix for the
  observed failure where VoxCPM spoke tag words aloud.
- Control strings are kept short (≤ ~4 words) because VoxCPM favors short control and long
  descriptors degrade output.

## Limitations (v1, documented intentionally)

- **Emotion fidelity is best-effort.** VoxCPM controllability is coarse and run-to-run
  inconsistent (per its README and paper); `[fear]`, `[dread]`, `[terror]` may sound alike.
  The interpreter delivers the model's native control faithfully — it cannot exceed the
  model's ceiling.
- **`[laughs]` and other non-verbal sounds are not synthesized.** They are stripped. A future
  version may insert pre-recorded sound clips (same mechanism as silence insertion).
- **Per-segment generation is slower** than one-shot, because each control change is a
  separate `generate` call. This is inherent to per-line emotion control.

## Testing strategy (all GPU-free)

Parser unit tests (`tests/test_audio_tags.py`):
- single emotion tag → one speech segment with expected control
- emotion persists across untagged text until next emotion tag
- pace + energy compose into the control string in order
- `[short pause]` / `[long pause]` emit silence segments of 0.30 / 0.70 s
- unknown tag is stripped, warns, and never appears in any segment text
- `[laughs]` is stripped and emits nothing
- adjacent identical-control text merges into one segment
- no-tags input → single empty-control speech segment

Driver unit test (fake `generate` returning a fixed-length array):
- silence segments insert `int(duration*sample_rate)` zero samples
- each speech segment is generated with the correct `(control)text` string
- output is the concatenation of all segment arrays in order

## Files

- Create: `src/voxcpm/audio_tags.py`
- Modify: `src/voxcpm/core.py` (add `generate_from_tagged_script`)
- Create: `tests/test_audio_tags.py`
