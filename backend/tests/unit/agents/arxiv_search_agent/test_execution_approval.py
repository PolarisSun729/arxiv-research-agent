from agents.arxiv_search_agent.execution.approvals import arguments_fingerprint


def test_arguments_fingerprint_is_stable_for_equivalent_mappings() -> None:
    first = {"paper": {"arxiv_id": "2401.00001", "title": "Example"}, "force": False}
    second = {"force": False, "paper": {"title": "Example", "arxiv_id": "2401.00001"}}

    assert arguments_fingerprint(first) == arguments_fingerprint(second)


def test_arguments_fingerprint_changes_when_approved_arguments_change() -> None:
    original = {"paper": {"arxiv_id": "2401.00001"}, "force": False}
    changed = {"paper": {"arxiv_id": "2401.00002"}, "force": False}

    assert arguments_fingerprint(original) != arguments_fingerprint(changed)
