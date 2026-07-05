from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AT_PATH = ROOT / "src" / "voxcpm" / "audio_tags.py"

# Stub voxcpm package so imports work
pkg = types.ModuleType("voxcpm")
pkg.__path__ = [str(ROOT / "src" / "voxcpm")]
sys.modules.setdefault("voxcpm", pkg)

spec = importlib.util.spec_from_file_location("voxcpm.audio_tags", AT_PATH)
at = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules["voxcpm.audio_tags"] = at
spec.loader.exec_module(at)


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
