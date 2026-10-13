"""清除旧迁移遗留的 manual 研究画像，并重新合并 effective 画像。

背景：
    早期画像迁移把 n-gram 抽取出的系统主题写进了 user_manual_profiles，source 标记为
    legacy_migration_low_confidence。manual 层本应只保存用户的显式操作，这批数据会以
    「手动设置」的高优先级覆盖自动画像，导致同一主题同时出现在关注和排除里。

用法（任意目录运行均可）：
    python scripts/clear_legacy_manual_profiles.py --dry-run   # 只打印将被删除的内容
    python scripts/clear_legacy_manual_profiles.py             # 实际删除并刷新 effective 画像

执行前请先备份 SQLite 数据库（默认 backend/06-database/recommendation.db）。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))

LEGACY_SOURCE = "legacy_migration_low_confidence"


def main() -> int:
    parser = argparse.ArgumentParser(description="清除 source=legacy_migration_low_confidence 的 manual 研究画像")
    parser.add_argument("--dry-run", action="store_true", help="只打印将被删除的行，不修改数据库")
    args = parser.parse_args()

    from dotenv import load_dotenv

    repo_root = Path(__file__).resolve().parents[1]
    load_dotenv(repo_root / ".env", override=False)
    load_dotenv(repo_root / "backend" / ".env", override=False)
    # 后端模块在加载环境变量之后再导入，保证数据库路径来自上面的配置。
    from services.storage.sqlite import StorageContainer

    storage = StorageContainer()
    store = storage.research_profiles
    print(f"数据库：{storage.connection_provider.db_path}")
    with storage.connection_provider.connect() as conn:
        rows = conn.execute(
            "SELECT user_id, profile_json, updated_at FROM user_manual_profiles WHERE source = ?",
            (LEGACY_SOURCE,),
        ).fetchall()
        if not rows:
            print("没有需要清除的旧迁移 manual 画像。")
            return 0
        for user_id, profile_json, updated_at in rows:
            profile = json.loads(profile_json or "{}")
            print(f"- user_id={user_id} updated_at={updated_at}")
            for field in ("positive_topics", "negative_topics", "recent_topics", "pinned_topics", "hidden_topics"):
                if profile.get(field):
                    print(f"    {field}: {profile[field]}")
        if args.dry_run:
            print(f"[dry-run] 将删除 {len(rows)} 行，未修改数据库。")
            return 0
        for user_id, _profile_json, _updated_at in rows:
            conn.execute("DELETE FROM user_manual_profiles WHERE user_id = ? AND source = ?", (user_id, LEGACY_SOURCE))
            # manual 层被删除后必须立刻重新合并，否则 effective 画像仍是旧的并集结果。
            store._refresh_effective_profile(conn, user_id)
        conn.commit()
    print(f"已删除 {len(rows)} 行旧迁移 manual 画像，并刷新对应用户的 effective 画像。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
