"""从逐次运行快照建立可比较配置，缺失版本不能被相同的空值伪装成固定环境。"""

import hashlib
import json
from typing import Any


def _missing_configuration(run: dict[str, Any]) -> list[str]:
    configuration = run.get("configuration") or {}
    engine = configuration.get("engine") or {}
    missing = []
    if not configuration.get("research_limits"):
        missing.append("research_limits")
    generation = engine.get("generation") or {}
    models = generation.get("models") or {}
    if not generation.get("provider") or not models or any(not value or value == "unknown" for value in models.values()):
        missing.append("generation_models")
    if not engine.get("retrieval"):
        missing.append("retrieval_configuration")
    indexes = (run.get("paper_context") or {}).get("indexes") or []
    if len(indexes) != 1:
        # 一次运行跨活动索引版本无法形成固定基线；零次快照也不能推断索引未变。
        missing.append("index_snapshot_missing_or_changed")
    else:
        index = indexes[0]
        build_id = str(index.get("active_build_id") or "")
        version = str(index.get("active_index_version") or "")
        immutable_build = bool(build_id and version and not build_id.startswith("legacy") and version != "legacy")
        if not index.get("collection_name") or not (immutable_build or index.get("sparse_index_source_hash")):
            missing.append("index_version_or_source_hash")
        if not index.get("embedding_model"):
            missing.append("embedding_model")
    return missing


def configuration_metadata(case_records: list[dict[str, Any]]) -> dict[str, Any]:
    snapshots, issues = {}, {}
    for record in case_records:
        case_id = record["golden_case"]["case_id"]
        runs = record.get("raw_runs") or []
        case_issues = set()
        versions = {}
        for run in runs:
            case_issues.update(_missing_configuration(run))
            snapshot = {"configuration": run.get("configuration"), "paper_context": run.get("paper_context")}
            canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            versions[canonical] = snapshot
        if not runs:
            case_issues.add("run_configuration_missing")
        if len(versions) > 1:
            case_issues.add("configuration_changed_between_repeats")
        if case_issues:
            issues[case_id] = sorted(case_issues)
        snapshots[case_id] = list(versions.values())
    fingerprint = None
    if snapshots and not issues:
        fingerprint = hashlib.sha256(json.dumps(snapshots, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    return {"evaluation_configuration": snapshots, "evaluation_config_hash": fingerprint, "configuration_issues": issues}
