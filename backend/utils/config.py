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

RETRIEVAL_CONFIG = {
    "default_top_k": 5,
    "candidate_multiplier": 3,
    "enable_query_rewrite": True,
    "enable_hyde": True,
    "enable_keyword_search": True,
    "debug": False,
    "rrf_k": 60,
    "route_weights": {
        "vector_original": 1.0,
        "vector_rewrite": 0.9,
        "vector_hyde": 0.85,
        "keyword": 0.75,
    },
}
