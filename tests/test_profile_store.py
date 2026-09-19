import json

import pytest

from src.core.models import (
    AntiPatterns,
    CadenceSyntax,
    DeepStyleProfile,
    DiscourseArchitecture,
    LexiconRhetoric,
    StyleProfile,
    TonePersona,
)
from src.memory.profile_store import ProfileStore


def make_profile(profile_id: str, name: str) -> DeepStyleProfile:
    return DeepStyleProfile(
        name=name,
        profile_id=profile_id,
        qualitative=StyleProfile(
            tone_persona=TonePersona(perspective="第一人称", emotional_tone="冷静直接"),
            cadence_syntax=CadenceSyntax(sentence_style="长短句交替", paragraph_habit="紧凑"),
            lexicon_rhetoric=LexiconRhetoric(metaphor_style="具体", vocabulary_richness="丰富"),
            discourse=DiscourseArchitecture(opening_hook="设问", body_progression="递进", ending_style="留白"),
            anti_patterns=AntiPatterns(),
        ),
    )


def test_profile_store_round_trip_and_active_switch(tmp_path):
    storage_path = tmp_path / "profiles.json"
    store = ProfileStore(str(storage_path))
    alpha = make_profile("alpha", "Alpha")
    beta = make_profile("beta", "Beta")

    store.save(alpha)
    store.save(beta)

    reloaded = ProfileStore(str(storage_path))
    assert [p.profile_id for p in reloaded.list_profiles()] == ["alpha", "beta"]
    assert reloaded.get_active().profile_id == "beta"
    assert reloaded.set_active("alpha").name == "Alpha"
    assert ProfileStore(str(storage_path)).get_active().profile_id == "alpha"


def test_interleaved_profile_store_instances_do_not_lose_updates(tmp_path):
    storage_path = str(tmp_path / "profiles.json")
    first = ProfileStore(storage_path)
    second = ProfileStore(storage_path)

    first.save(make_profile("alpha", "Alpha"))
    second.save(make_profile("beta", "Beta"))

    assert {p.profile_id for p in first.list_profiles()} == {"alpha", "beta"}
    assert first.get_active().profile_id == "beta"


def test_profile_store_rejects_missing_id_and_preserves_corrupt_file(tmp_path):
    storage_path = tmp_path / "profiles.json"
    store = ProfileStore(str(storage_path))
    with pytest.raises(ValueError, match="profile_id"):
        store.save(make_profile("alpha", "Alpha").model_copy(update={"profile_id": None}))

    storage_path.write_text("{broken json", encoding="utf-8")
    with pytest.raises(json.JSONDecodeError):
        store.list_profiles()
    with pytest.raises(json.JSONDecodeError):
        store.save(make_profile("beta", "Beta"))
    assert storage_path.read_text(encoding="utf-8") == "{broken json"
