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
