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
