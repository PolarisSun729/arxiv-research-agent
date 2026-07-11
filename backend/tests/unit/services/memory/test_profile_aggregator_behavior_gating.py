from services.memory.profile_aggregator import ProfileAggregator


def _paper(arxiv_id: str, concept: str, *, categories=None):
    return {
        "arxiv_id": arxiv_id,
        "categories": categories or ["cs.AI"],
        "evidence_card": {
            "schema_valid": True,
            "candidate_concepts": [
                {
                    "label": concept,
                    "type": "technical_concept",
                    "confidence": 0.9,
                    "evidence_text": "test evidence",
                    "source": "llm",
                    "whether_generalizable": True,
                }
            ],
            "extraction_confidence": 0.9,
        },
    }


def test_weak_liked_paper_does_not_become_long_term_positive_topic():
    result = ProfileAggregator().aggregate(
        {
            "liked_papers": [_paper("2401.00001", "outlier unique topic")],
            "interest_model": {
                "status": "insufficient",
                "stable_positive_paper_ids": [],
                "weak_positive_paper_ids": ["2401.00001"],
                "stable_negative_paper_ids": [],
                "weak_negative_paper_ids": [],
                "profile_gating": {
                    "positive_min_topic_source_papers": 2,
                    "negative_min_topic_source_papers": 2,
                },
            },
        },
        current_profile={},
        preserve_existing_topics=False,
        preserve_existing_representative_papers=False,
    )

    assert "outlier unique topic" not in result["positive_topics"]
    assert "outlier unique topic" in result["recent_topics"]
    assert result["preferred_categories"] == []
    assert result["aggregation_report"]["behavior_profile_gating"]["skipped_positive_papers"] == ["2401.00001"]


def test_only_stable_cluster_papers_contribute_long_term_positive_topics():
    result = ProfileAggregator().aggregate(
        {
            "liked_papers": [
                _paper("2401.00001", "stable shared topic", categories=["cs.CL"]),
                _paper("2401.00002", "stable shared topic", categories=["cs.CL"]),
                _paper("2401.00003", "outlier unique topic", categories=["cs.CV"]),
            ],
            "interest_model": {
                "status": "stable",
                "stable_positive_paper_ids": ["2401.00001", "2401.00002"],
                "weak_positive_paper_ids": ["2401.00003"],
                "stable_negative_paper_ids": [],
                "weak_negative_paper_ids": [],
                "profile_gating": {
                    "positive_min_topic_source_papers": 2,
                    "negative_min_topic_source_papers": 2,
                },
            },
        },
        current_profile={},
        preserve_existing_topics=False,
        preserve_existing_representative_papers=False,
    )

    assert "stable shared topic" in result["positive_topics"]
    assert "outlier unique topic" not in result["positive_topics"]
    assert result["preferred_categories"] == ["cs.CL"]
    assert "cs.CV" not in result["preferred_categories"]


def test_stable_cluster_topic_still_needs_multiple_source_papers():
    result = ProfileAggregator().aggregate(
        {
            "liked_papers": [
                _paper("2401.00001", "stable shared topic"),
                _paper("2401.00002", "stable shared topic"),
                _paper("2401.00003", "single source side topic"),
            ],
            "interest_model": {
                "status": "stable",
                "stable_positive_paper_ids": ["2401.00001", "2401.00002", "2401.00003"],
                "weak_positive_paper_ids": [],
                "stable_negative_paper_ids": [],
                "weak_negative_paper_ids": [],
                "profile_gating": {
                    "positive_min_topic_source_papers": 2,
                    "negative_min_topic_source_papers": 2,
                },
            },
        },
        current_profile={},
        preserve_existing_topics=False,
        preserve_existing_representative_papers=False,
    )

    assert "stable shared topic" in result["positive_topics"]
    assert "single source side topic" not in result["positive_topics"]
    filtered = result["aggregation_report"]["behavior_profile_gating"]["filtered_topics"]
    assert any(
        item["topic"] == "single source side topic"
        and item["field"] == "positive_topics"
        and item["source_paper_count"] == 1
        for item in filtered
    )


def test_weak_disliked_paper_does_not_become_long_term_negative_topic():
    result = ProfileAggregator().aggregate(
        {
            "disliked_papers": [_paper("2401.00004", "unstable negative topic")],
            "interest_model": {
                "status": "insufficient",
                "stable_positive_paper_ids": [],
                "weak_positive_paper_ids": [],
                "stable_negative_paper_ids": [],
                "weak_negative_paper_ids": ["2401.00004"],
                "profile_gating": {
                    "positive_min_topic_source_papers": 2,
                    "negative_min_topic_source_papers": 2,
                },
            },
        },
        current_profile={},
        preserve_existing_topics=False,
        preserve_existing_representative_papers=False,
    )

    assert "unstable negative topic" not in result["negative_topics"]
    assert result["aggregation_report"]["behavior_profile_gating"]["skipped_negative_papers"] == ["2401.00004"]
