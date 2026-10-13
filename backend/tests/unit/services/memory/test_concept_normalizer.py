import pytest

from services.memory.concept_normalizer import ConceptNormalizer


@pytest.mark.parametrize("label", ["coverage analysis", "storage efficiency", "average precision", "leverage points"])
def test_rule_patterns_do_not_match_inside_other_words(label):
    assert ConceptNormalizer()._match_rule(label) is None


@pytest.mark.parametrize(
    ("label", "expected"),
    [
        ("RAG pipelines", "RAG retrieval optimization"),
        ("retrieval-augmented generation", "RAG retrieval optimization"),
        ("diffusion models", "diffusion models"),
        ("open-domain question answering", "question answering"),
    ],
)
def test_rule_patterns_still_match_whole_words(label, expected):
    assert ConceptNormalizer()._match_rule(label)["label"] == expected


def test_topic_key_folds_plurals():
    assert ConceptNormalizer._topic_key("large language models") == ConceptNormalizer._topic_key("large language model")


def test_cluster_signals_groups_all_polarities_into_one_topic():
    signals = [
        {"label": "retrieval-augmented generation", "confidence": 0.9, "polarity": 1},
        {"label": "RAG", "confidence": 0.9, "polarity": -1},
    ]
    topics = ConceptNormalizer().cluster_signals(signals)

    assert [topic["label"] for topic in topics] == ["RAG retrieval optimization"]
    assert {member["polarity"] for member in topics[0]["members"]} == {1, -1}


def _signal(label, paper, confidence=0.9):
    return {"label": label, "confidence": confidence, "polarity": 1, "source_papers": [paper]}


def test_same_concept_with_different_casing_lands_in_one_topic():
    signals = [_signal("Large Language Models", "p1"), _signal("Factuality Enhancement in Large Language Models", "p2"), _signal("large language model", "p3")]
    topics = ConceptNormalizer().cluster_signals(signals)
    owners = [topic["label"] for topic in topics if any(m["label"].lower().startswith("large language model") for m in topic["members"])]

    assert len(owners) == 1


def test_loose_clusters_do_not_chain_through_intermediate_concepts():
    # A≈B、B≈C 但 A 与 C 不相似：只和簇种子比较时，C 不会被 B 串进 A 的簇。
    signals = [
        _signal("graph neural retrieval", "p1"),
        _signal("graph neural retrieval", "p2"),
        _signal("graph neural networks", "p3"),
        _signal("convolutional neural networks", "p4"),
    ]
    topics = ConceptNormalizer().cluster_signals(signals)
    by_member = {member["label"]: topic["label"] for topic in topics for member in topic["members"]}

    assert by_member["convolutional neural networks"] != by_member["graph neural retrieval"]


def test_cluster_label_prefers_concept_supported_by_most_papers():
    signals = [
        _signal("sparse attention kernels", "p1", confidence=0.6),
        _signal("sparse attention kernels", "p2", confidence=0.6),
        _signal("efficient sparse attention kernels", "p3", confidence=0.95),
    ]
    topics = ConceptNormalizer().cluster_signals(signals)

    assert [topic["label"] for topic in topics] == ["sparse attention kernels"]
