from datetime import datetime, timezone

from services.memory.profile_aggregator import ProfileAggregator


NOW = datetime(2026, 10, 1, tzinfo=timezone.utc)


def _paper(arxiv_id: str, concepts, *, at: str):
    return {
        "arxiv_id": arxiv_id,
        "categories": ["cs.CL"],
        "liked_at": at,
        "disliked_at": at,
        "evidence_card": {
            "schema_valid": True,
            "extraction_confidence": 0.9,
            "candidate_concepts": [
                {"label": concept, "type": "technical_concept", "confidence": 0.9, "source": "llm", "whether_generalizable": True}
                for concept in concepts
            ],
        },
    }


def _aggregate(evidence):
    return ProfileAggregator().aggregate(evidence, now=NOW)


def test_concept_seen_in_both_liked_and_disliked_papers_enters_no_zone():
    # 过泛概念不靠规则识别：它同时出现在喜欢和不喜欢的论文里，正负一致性低，自然进不了任何区。
    liked = [_paper(f"2601.0000{i}", ["large language models", "graph retrieval"], at=f"2026-08-0{i} 00:00:00") for i in range(1, 4)]
    liked[-1]["liked_at"] = "2026-08-20 00:00:00"
    disliked = [_paper(f"2601.0001{i}", ["large language models", "speech synthesis"], at=f"2026-08-0{i} 00:00:00") for i in range(1, 4)]
    result = _aggregate({"liked_papers": liked, "disliked_papers": disliked})

    assert "large language models" not in result["positive_topics"]
    assert "large language models" not in result["negative_topics"]
    assert "mixed_positive_negative_evidence" in result["topic_evidence"]["large language models"]["rejection_reasons"]
    assert "graph retrieval" in result["positive_topics"]
    assert "speech synthesis" in result["negative_topics"]


def test_persistent_interest_becomes_long_term_but_single_day_burst_only_recent():
    persistent = [
        _paper("2601.00001", ["graph retrieval"], at="2026-08-01 00:00:00"),
        _paper("2601.00002", ["graph retrieval"], at="2026-09-20 00:00:00"),
    ]
    burst = [
        _paper("2601.00003", ["agent memory"], at="2026-09-30 10:00:00"),
        _paper("2601.00004", ["agent memory"], at="2026-09-30 11:00:00"),
        _paper("2601.00005", ["agent memory"], at="2026-09-30 12:00:00"),
    ]
    result = _aggregate({"liked_papers": [*persistent, *burst]})

    assert "graph retrieval" in result["positive_topics"]
    assert "agent memory" not in result["positive_topics"]
    assert "agent memory" in result["recent_topics"]
    assert "not_persistent_enough" in result["topic_evidence"]["agent memory"]["rejection_reasons"]


def test_recent_score_decays_with_time():
    old = _aggregate({"liked_papers": [_paper(f"2601.0000{i}", ["graph retrieval"], at="2026-06-01 00:00:00") for i in (1, 2)]})
    fresh = _aggregate({"liked_papers": [_paper(f"2601.0000{i}", ["graph retrieval"], at="2026-09-30 00:00:00") for i in (1, 2)]})

    assert "graph retrieval" not in old["recent_topics"]
    assert "graph retrieval" in fresh["recent_topics"]
    assert old["topic_evidence"]["graph retrieval"]["long_pos"] < fresh["topic_evidence"]["graph retrieval"]["long_pos"]


def test_focus_and_exclude_are_mutually_exclusive_and_exclude_never_recent():
    liked = [
        _paper("2601.00001", ["graph retrieval", "speech synthesis"], at="2026-08-01 00:00:00"),
        _paper("2601.00002", ["graph retrieval"], at="2026-09-29 00:00:00"),
    ]
    disliked = [_paper(f"2601.0001{i}", ["speech synthesis"], at="2026-09-29 00:00:00") for i in range(1, 6)]
    result = _aggregate({"liked_papers": liked, "disliked_papers": disliked})

    assert not set(result["positive_topics"]) & set(result["negative_topics"])
    assert not set(result["recent_topics"]) & set(result["negative_topics"])
    assert "speech synthesis" in result["negative_topics"]
    assert all(result["topic_evidence"][topic]["net_score"] > 0 for topic in result["positive_topics"])
    assert all(result["topic_evidence"][topic]["net_score"] < 0 for topic in result["negative_topics"])


def test_like_counted_once_even_when_present_in_events_and_preference_table():
    event = {"event_id": "evt-1", "event_type": "liked", "created_at": "2026-09-30 00:00:00"}
    paper = {**_paper("2601.00001", ["graph retrieval"], at="2026-09-30 00:00:00"), "_profile_event": event}
    once = _aggregate({"liked_papers": [paper]})
    twice = _aggregate(
        {
            "liked_papers": [paper],
            "recent_actions": [{"arxiv_id": "2601.00001", "action_type": "liked", "paper": paper, "_profile_event": event}],
        }
    )

    assert once["topic_evidence"]["graph retrieval"]["long_pos"] == twice["topic_evidence"]["graph retrieval"]["long_pos"]


def test_note_tag_is_explicit_and_skips_persistence_requirement():
    note = {"note_id": "n1", "arxiv_id": "2601.00009", "include_in_profile": True, "tags": ["sparse attention"], "created_at": "2026-09-30 00:00:00"}
    result = _aggregate({"notes": [note]})

    assert "sparse attention" in result["positive_topics"]
    assert result["topic_evidence"]["sparse attention"]["source_notes"] == ["n1"]


def test_single_paper_concepts_do_not_fill_recent_zone():
    # 一篇论文上的问答会带出一串专有概念；近期区要求至少两篇论文支撑，它们都不该进入。
    qa_paper = _paper("2601.00001", ["PEC2F Graph Schema", "Epistemic Separation", "graph retrieval"], at="2026-09-30 00:00:00")
    other = _paper("2601.00002", ["graph retrieval"], at="2026-09-29 00:00:00")
    actions = [
        {"arxiv_id": paper["arxiv_id"], "action_type": "qa_asked", "paper": paper, "created_at": "2026-09-30 00:00:00"}
        for paper in (qa_paper, other)
    ]
    result = _aggregate({"recent_actions": actions})

    assert result["recent_topics"] == ["graph retrieval"]
    assert "insufficient_recent_papers" in result["topic_evidence"]["PEC2F Graph Schema"]["rejection_reasons"]


def test_same_paper_counted_once_per_topic_across_card_fields():
    single = _paper("2601.00001", ["retrieval-augmented generation"], at="2026-09-30 00:00:00")
    repeated = _paper("2601.00001", ["retrieval-augmented generation", "RAG pipelines"], at="2026-09-30 00:00:00")
    repeated["evidence_card"]["tasks"] = ["Retrieval-Augmented Generation"]
    once = _aggregate({"liked_papers": [single]})["topic_evidence"]["RAG retrieval optimization"]
    many = _aggregate({"liked_papers": [repeated]})["topic_evidence"]["RAG retrieval optimization"]

    assert once["long_pos"] == many["long_pos"]


def test_coarse_card_fields_link_papers_with_disjoint_fine_concepts():
    first = _paper("2601.00001", ["Entity Semantic Restructuring"], at="2026-08-01 00:00:00")
    second = _paper("2601.00002", ["Topic Consistency Filtering"], at="2026-09-20 00:00:00")
    first["evidence_card"]["tasks"] = ["Knowledge Graph Construction"]
    second["evidence_card"]["main_research_area"] = "knowledge graph construction"
    result = _aggregate({"liked_papers": [first, second]})

    assert "knowledge graph construction" in result["positive_topics"]
    assert result["topic_evidence"]["knowledge graph construction"]["liked_papers"] == 2


def test_generic_concept_with_mixed_history_does_not_enter_recent_zone():
    liked = [_paper(f"2601.0000{i}", ["large language models"], at="2026-07-0{i} 00:00:00") for i in range(1, 4)]
    disliked = [_paper(f"2601.0001{i}", ["large language models"], at="2026-07-0{i} 00:00:00") for i in range(1, 4)]
    asked = [_paper(f"2601.0002{i}", ["large language models"], at="2026-09-30 00:00:00") for i in range(1, 3)]
    actions = [{"arxiv_id": paper["arxiv_id"], "action_type": "qa_asked", "paper": paper, "created_at": "2026-09-30 00:00:00"} for paper in asked]
    result = _aggregate({"liked_papers": liked, "disliked_papers": disliked, "recent_actions": actions})

    assert "large language models" not in result["recent_topics"]
    assert result["topic_evidence"]["large language models"]["recent_net"] >= 0.6
