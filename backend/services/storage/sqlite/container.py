from services.storage.sqlite.connection import SqliteConnectionProvider
from services.storage.sqlite.schema import StorageSchemaMigrator
from services.storage.sqlite.stores import (
    AgentSessionStore,
    AgentRuntimeCheckpointStore,
    InterestVectorStore,
    LangGraphCheckpointStore,
    PaperCatalogStore,
    PaperChatMessageStore,
    PaperChatSessionStore,
    PaperNoteStore,
    PaperProfileEvidenceStore,
    PaperQAIndexStore,
    PaperQATurnStore,
    ProfileBuildJobStore,
    ProfileEventStore,
    ResearchProfileStore,
    UserPreferenceStore,
)


class StorageContainer:
    """SQLite 存储组合根，只负责初始化和装配，不作为业务万能入口。"""

    def __init__(
        self,
        *,
        connection_provider: SqliteConnectionProvider | None = None,
        initialize_schema: bool = True,
    ) -> None:
        self.connection_provider = connection_provider or SqliteConnectionProvider()
        self.schema_migrator = StorageSchemaMigrator(self.connection_provider)
        if initialize_schema:
            # schema 初始化集中在组合根执行，避免任意 store 构造时偷偷触发全库迁移。
            self.schema_migrator.ensure_schema()

        self.profile_events = ProfileEventStore(self.connection_provider)
        self.profile_build_jobs = ProfileBuildJobStore(self.connection_provider)
        self.research_profiles = ResearchProfileStore(
            self.connection_provider,
            self.profile_events,
            self.profile_build_jobs,
        )
        if initialize_schema:
            # legacy profile 数据迁移依赖画像 store 的标准化逻辑，schema 就绪后显式执行。
            self.research_profiles.migrate_legacy_profiles()
        self.paper_catalog = PaperCatalogStore(self.connection_provider)
        self.user_preferences = UserPreferenceStore(
            self.connection_provider,
            self.profile_events,
            self.research_profiles,
        )
        self.interest_vectors = InterestVectorStore(self.connection_provider)
        self.paper_profile_evidence = PaperProfileEvidenceStore(self.connection_provider)
        self.paper_qa_index = PaperQAIndexStore(self.connection_provider)
        self.paper_chat_sessions = PaperChatSessionStore(self.connection_provider)
        self.paper_chat_messages = PaperChatMessageStore(
            self.connection_provider,
            self.paper_chat_sessions,
            self.profile_events,
        )
        self.paper_qa_turns = PaperQATurnStore(
            self.connection_provider,
            self.paper_chat_sessions,
            self.profile_events,
        )
        self.paper_notes = PaperNoteStore(self.connection_provider, self.profile_events)
        self.agent_sessions = AgentSessionStore(self.connection_provider)
        self.agent_runtime_checkpoints = AgentRuntimeCheckpointStore(self.connection_provider)
        self.langgraph_checkpoints = LangGraphCheckpointStore(self.connection_provider)
