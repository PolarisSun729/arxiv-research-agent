from services.storage.sqlite.stores.research_profiles import ResearchProfileStore


def _generated():
    def _canonical(label, aliases=()):
        return {"label": label, "aliases": list(aliases), "net_score": 1.0}

    return {
        "positive_topics": ["vision-language models", "large language model"],
        "negative_topics": ["LLM reasoning"],
        "recent_topics": ["vision-language models"],
        "canonical_topics": [_canonical("vision-language models"), _canonical("large language model")],
        "canonical_negative_topics": [_canonical("LLM reasoning", ["chain-of-thought"])],
        "canonical_recent_topics": [_canonical("vision-language models")],
        "topic_evidence": {
            "vision-language models": {"label": "vision-language models", "provenance": "generated", "placement": ["focus", "recent"], "net_score": 8.28},
            "LLM reasoning": {"label": "LLM reasoning", "provenance": "generated", "placement": ["exclude"], "net_score": -8.976},
        },
    }


def _merge(manual):
    return ResearchProfileStore._merge_profile_projection("u1", _generated(), manual)


def _assert_exclusive(profile):
    assert not set(profile["positive_topics"]) & set(profile["negative_topics"])
    assert not set(profile["recent_topics"]) & set(profile["negative_topics"])


def test_manual_exclude_overrides_generated_focus():
    profile = _merge({"negative_topics": ["vision-language models"]})

    _assert_exclusive(profile)
    assert "vision-language models" in profile["negative_topics"]
    assert "vision-language models" not in profile["positive_topics"]
    evidence = profile["topic_evidence"]["vision-language models"]
    assert evidence["provenance"] == "manual"
    assert evidence["overridden_by_manual"] is True
    assert evidence["generated_evidence"]["net_score"] == 8.28
    assert profile["manual_overrides"][0]["topic"] == "vision-language models"


def test_manual_focus_overrides_generated_exclude():
    profile = _merge({"positive_topics": ["LLM reasoning"]})

    _assert_exclusive(profile)
    assert "LLM reasoning" in profile["positive_topics"]
    assert "LLM reasoning" not in profile["negative_topics"]
    assert profile["topic_evidence"]["LLM reasoning"]["manual_action"] == "focus"


def test_hidden_topic_removed_from_every_zone():
    profile = _merge({"hidden_topics": ["vision-language models"]})

    assert "vision-language models" not in profile["positive_topics"] + profile["negative_topics"] + profile["recent_topics"]


def test_manual_topic_matches_generated_label_by_plural_and_alias():
    profile = _merge({"negative_topics": ["large language models"], "positive_topics": ["chain-of-thought"]})

    _assert_exclusive(profile)
    assert "large language model" in profile["negative_topics"]
    assert "large language model" not in profile["positive_topics"]
    assert "LLM reasoning" in profile["positive_topics"]
    assert "chain-of-thought" not in profile["positive_topics"]


def test_pinned_topics_come_first():
    profile = _merge({"pinned_topics": ["graph retrieval"], "positive_topics": ["sparse attention"]})

    assert profile["positive_topics"][:2] == ["graph retrieval", "sparse attention"]
    assert profile["canonical_topics"][0]["label"] == "graph retrieval"
    assert profile["canonical_topics"][0]["pinned"] is True
