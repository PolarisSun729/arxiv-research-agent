from services.storage.sqlite.connection import SqliteConnectionProvider
from services.storage.sqlite.serialization import JsonFieldCodec


class BaseSqliteStore(JsonFieldCodec):
    """业务 store 的共同底座，只提供连接和 JSON 编解码，不承载业务入口。"""

    def __init__(self, connection_provider: SqliteConnectionProvider) -> None:
        self.connection_provider = connection_provider

    def _get_connection(self):
        return self.connection_provider.connect()
