# VoxCPM2 Colab Notebook — Narration Workflow Upgrade

**Date:** 2026-06-26
**File touched:** `VoxCPM2_Gradio_Colab_AIQUEST.ipynb` (cell 4 = Gradio app, plus one new Drive-mount cell)
**Not touched:** `app.py` — the Colab notebook is a self-contained inline Gradio app and does not import `app.py`.

## Problem

User generates YouTube story-video voiceovers on Google Colab via the notebook's `gradio.live` link. Two pain points:

1. **Long script errors.** Short scripts generate fine, but pasting a whole story at once fails. The notebook's `voice_clone()` passes the full text straight to `model.generate()` with no chunking.
2. **Voice presets do not persist.** The reference voice must be re-uploaded every time. Colab local disk is wiped on every runtime disconnect, and there is no save/preset mechanism.

Both fixes land in the **Voice Cloning tab** (the tab the user uses for narration). The other three tabs (Text-to-Speech, Voice Design, Ultimate Cloning) stay unchanged.

## Solution

### 1. Long-script chunking (Voice Cloning tab)

Port the chunking approach already proven in `app.py::split_text_into_chunks`:

- Split target text by paragraph (`\n`), then by sentence terminators (`.`, `!`, `?`, `。`, `！`, `？`, `।`), capping each chunk at ~150 chars. Oversized single sentences fall back to comma/semicolon splits.
- Generate each chunk with the **same reference voice** so timbre stays consistent.
- Insert a **0.3s silence** between paragraph boundaries for natural narration pacing.
- `np.concatenate` all chunk waveforms into **one final WAV**.
- Show a Gradio progress bar across chunks (e.g. "chunk 3/12").

Helper functions added to cell 4:
- `split_text_into_chunks(text, max_chars=150) -> list[str]` — ported from `app.py`.
- A silence array of `0.3 * SAMPLE_RATE` zeros inserted between chunks.

`voice_clone()` is rewritten to loop over chunks, collect per-chunk audio via the existing `collect_audio()` generator helper, interleave silence, and concatenate.

### 2. Google Drive voice presets

**New cell** (placed after GPU verify, before/with model load): mount Drive and ensure the presets folder.

```python
from google.colab import drive
drive.mount('/content/drive')
VOICES_DIR = '/content/drive/MyDrive/VoxCPM_Voices'
os.makedirs(VOICES_DIR, exist_ok=True)
```

**Voice Cloning tab UI changes:**
- Add a **"Saved Voice" dropdown** listing `*.wav` files in `VOICES_DIR`, plus an "⬆️ Upload new" entry.
- Keep the existing **reference-audio upload** widget (shown when "Upload new" is selected).
- Add a **"Preset name" textbox** + **"💾 Save Voice" button**: copies the currently uploaded audio into `VOICES_DIR/<clean_name>.wav` (sanitize name like `app.py::save_voice` does), then refreshes the dropdown and selects the new preset.
- When a saved preset is selected, `voice_clone()` resolves the reference path to `VOICES_DIR/<name>.wav` instead of the upload widget — no re-upload needed.

**Cross-session behavior:** mounting Drive at the start of any new session immediately exposes all previously saved narrator voices in the dropdown.

### 3. Scope guardrails (YAGNI)

- Only the Voice Cloning tab and one new Drive-mount cell change. Tabs 1, 2, 4 untouched.
- No new pip dependencies — `google.colab.drive`, `soundfile`, `numpy` are already present in Colab.
- No delete/rename UI for presets in this iteration (user can manage files in Drive directly).

## Data flow

```
Story text ──► split_text_into_chunks ──► [chunk1, chunk2, ...]
                                              │
ref voice (Drive preset OR upload) ───────────┤ per chunk
                                              ▼
                              model.generate(text=chunk, reference_wav_path=ref)
                                              ▼
                          collect_audio() ──► chunk_wav
                                              ▼
          concat( chunk1, silence(0.3s), chunk2, silence, ... ) ──► final WAV ──► Gradio output
```

## Error handling

- Empty text / no reference selected → `gr.Error` (matches existing pattern).
- If a saved preset is selected but the dropdown value isn't a real file → `gr.Error("Saved voice not found, re-select or upload.")`.
- Save with empty name or no uploaded audio → `gr.Error`.
- Drive mount failure surfaces Colab's own auth error (out of scope to wrap).

## Success criteria

1. Pasting a full multi-paragraph story into the Voice Cloning tab produces one continuous WAV with no error, narrated in the cloned voice, with brief pauses between paragraphs.
2. Uploading a voice + saving it puts a `.wav` in `MyDrive/VoxCPM_Voices/` and it appears in the dropdown.
3. Disconnecting the runtime, starting fresh, re-running cells, and re-mounting Drive shows the saved voice in the dropdown — selectable without re-upload.
4. Tabs 1, 2, 4 behave exactly as before.
