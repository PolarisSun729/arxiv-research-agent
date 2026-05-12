from enum import Enum
from typing import Dict, Any

class VectorDBProvider(str, Enum):
    MILVUS = "milvus"
    # More providers can be added later

class DatabaseType(str, Enum):
    SQLITE = "sqlite"

MILVUS_CONFIG = {
    "uri": "http://localhost:19530",
    "index_types": {
        "flat": "FLAT",
        "ivf_flat": "IVF_FLAT",
        "ivf_sq8": "IVF_SQ8",
        "hnsw": "HNSW"
    },
    "index_params": {
        "flat": {},
        "ivf_flat": {"nlist": 1024},
        "ivf_sq8": {"nlist": 1024},
        "hnsw": {
            "M": 16,
            "efConstruction": 500
        }
    }
} 

SQLITE_CONFIG = {
    "database_path": "06-database/recommendation.db",
    "check_same_thread": False
}