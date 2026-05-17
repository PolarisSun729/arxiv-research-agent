import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))


def print_route(title: str, retrieval_debug: dict):
    print(f"\n=== {title} ===")
    print(f"original_query: {retrieval_debug.get('original_query')}")
    if retrieval_debug.get("query_plan"):
        print(f"query_plan: {retrieval_debug.get('query_plan')}")
    print(f"rewritten_queries: {retrieval_debug.get('rewritten_queries')}")
    hyde_text = retrieval_debug.get("hyde_text") or ""
    print(f"hyde_text: {hyde_text[:180]}{'...' if len(hyde_text) > 180 else ''}")
    for route_name, items in retrieval_debug.get("routes", {}).items():
        print(f"\n[{route_name}]")
        if not items:
            print("  no hits")
            continue
        for idx, item in enumerate(items[:5], start=1):
            preview = item.get("preview", "")
            print(
                f"  {idx}. chunk={item.get('chunk_id')} page={item.get('page_number') or item.get('page_range')} "
                f"route_score={item.get('route_score')} preview={preview}"
            )
    print("\n[final_chunks]")
    for idx, item in enumerate(retrieval_debug.get("final_chunks", []), start=1):
        print(
            f"  {idx}. chunk={item.get('chunk_id')} page={item.get('page_number') or item.get('page_range')} "
            f"fused_score={item.get('score')} matched_routes={item.get('matched_routes')} preview={item.get('preview')}"
        )


def main():
    parser = argparse.ArgumentParser(description="Compare base vector retrieval and enhanced retrieval strategies.")
    parser.add_argument("--arxiv-id", required=True, help="Paper arXiv id with an existing QA index")
    parser.add_argument("--query", required=True, help="Question to test")
    parser.add_argument("--top-k", type=int, default=5, help="Final top-k chunks")
    args = parser.parse_args()

    from services.database_service import DatabaseService
    from services.embedding_service import EmbeddingService
    from services.enhanced_retrieval_service import EnhancedRetrievalService, RetrievalOptions
    from services.generation_service import GenerationService
    from services.vector_store_service import VectorStoreService

    db_service = DatabaseService()
    qa_index = db_service.get_paper_qa_index(args.arxiv_id)
    if not qa_index or qa_index.get("status") != "indexed":
        raise SystemExit(f"QA index not found or not ready for {args.arxiv_id}")

    retriever = EnhancedRetrievalService(
        embedding_service=EmbeddingService(),
        vector_store_service=VectorStoreService(),
        generation_service=GenerationService(),
    )

    scenarios = [
        (
            "vector_only",
            RetrievalOptions(
                top_k=args.top_k,
                enable_query_rewrite=False,
                enable_hyde=False,
                enable_keyword_search=False,
                debug=True,
            ),
        ),
        (
            "vector_plus_rewrite",
            RetrievalOptions(
                top_k=args.top_k,
                enable_query_rewrite=True,
                enable_hyde=False,
                enable_keyword_search=False,
                debug=True,
            ),
        ),
        (
            "vector_plus_hyde",
            RetrievalOptions(
                top_k=args.top_k,
                enable_query_rewrite=False,
                enable_hyde=True,
                enable_keyword_search=False,
                debug=True,
            ),
        ),
        (
            "hybrid_all",
            RetrievalOptions(
                top_k=args.top_k,
                enable_query_rewrite=True,
                enable_hyde=True,
                enable_keyword_search=True,
                debug=True,
            ),
        ),
    ]

    for title, options in scenarios:
        result = retriever.enhanced_retrieve(
            user_query=args.query,
            collection_name=qa_index["collection_name"],
            options=options,
        )
        print_route(title, result["debug"])


if __name__ == "__main__":
    main()
