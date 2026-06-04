import unittest

from services.arxiv.arxiv_query_builder import (
    ArxivSearchValidationError,
    build_arxiv_field_clause,
    build_arxiv_query_from_structured_params,
    build_arxiv_raw_query,
    combine_arxiv_clauses,
    normalize_text_value,
    quote_arxiv_text,
    validate_arxiv_search_request,
)


class ArxivQueryBuilderUnitTests(unittest.TestCase):
    def test_normalize_text_value_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_text_value("  rag\n systems \t today  "), "rag systems today")
        self.assertEqual(normalize_text_value(None), "")

    def test_quote_arxiv_text_quotes_special_values_and_preserves_existing_quotes(self) -> None:
        self.assertEqual(quote_arxiv_text("rag"), "rag")
        self.assertEqual(quote_arxiv_text("rag systems"), '"rag systems"')
        self.assertEqual(quote_arxiv_text('"exact phrase"'), '"exact phrase"')

    def test_build_arxiv_field_clause_handles_text_and_category_fields(self) -> None:
        self.assertEqual(build_arxiv_field_clause("ti", "graph rag"), 'ti:"graph rag"')
        self.assertEqual(build_arxiv_field_clause("cat", "cs.CL"), "cat:cs.CL")
        self.assertEqual(build_arxiv_field_clause("ti", "   "), "")

    def test_combine_arxiv_clauses_filters_empty_values(self) -> None:
        self.assertEqual(combine_arxiv_clauses(["", "ti:rag", ""], "AND"), "ti:rag")
        self.assertEqual(combine_arxiv_clauses(["ti:rag", "au:lewis"], "OR"), "ti:rag OR au:lewis")

    def test_validate_arxiv_search_request_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(ArxivSearchValidationError):
            validate_arxiv_search_request(
                search_query=None,
                id_list=None,
                max_results=10,
                start=0,
                sort_by="relevance",
                sort_order="descending",
            )

        with self.assertRaises(ArxivSearchValidationError):
            validate_arxiv_search_request(
                search_query="rag",
                id_list=None,
                max_results=0,
                start=0,
                sort_by="relevance",
                sort_order="descending",
            )

    def test_validate_arxiv_search_request_accepts_valid_inputs(self) -> None:
        validate_arxiv_search_request(
            search_query="rag",
            id_list=None,
            max_results=10,
            start=0,
            sort_by="relevance",
            sort_order="descending",
        )

    def test_build_arxiv_raw_query_appends_submitted_date_filter(self) -> None:
        payload = build_arxiv_raw_query(search_query="rag", submitted_days_ago=7)

        self.assertIn("(rag) AND submittedDate:[", payload["final_search_query"])
        self.assertTrue(payload["submitted_days_ago_applied"])
        self.assertEqual(payload["normalized_inputs"]["search_query"], "rag")

    def test_build_arxiv_raw_query_can_use_date_only_when_query_missing(self) -> None:
        payload = build_arxiv_raw_query(search_query=None, submitted_days_ago=3, append_date_when_query_missing=True)

        self.assertIsNotNone(payload["final_search_query"])
        self.assertTrue(payload["final_search_query"].startswith("submittedDate:["))

    def test_build_arxiv_query_from_structured_params_combines_fields(self) -> None:
        payload = build_arxiv_query_from_structured_params(
            query="rag",
            title_query="graph retrieval",
            categories=["cs.CL", "cs.IR"],
            field_operator="and",
            category_operator="or",
        )

        final_query = payload["final_search_query"]
        self.assertIn("all:rag", final_query)
        self.assertIn('ti:"graph retrieval"', final_query)
        self.assertIn("(cat:cs.CL OR cat:cs.IR)", final_query)
        self.assertEqual(payload["normalized_inputs"]["field_operator"], "AND")

    def test_build_arxiv_query_from_structured_params_rejects_invalid_operator(self) -> None:
        with self.assertRaises(ArxivSearchValidationError):
            build_arxiv_query_from_structured_params(query="rag", field_operator="XOR")


if __name__ == "__main__":
    unittest.main()
