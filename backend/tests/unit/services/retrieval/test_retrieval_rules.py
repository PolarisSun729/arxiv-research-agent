import unittest

from services.retrieval.retrieval_rules import RetrievalRules


class KeywordTokenizerTests(unittest.TestCase):
    def test_plural_and_singular_share_one_stem(self) -> None:
        self.assertEqual(
            RetrievalRules.tokenize_for_keyword_search("cats datasets distributions"),
            RetrievalRules.tokenize_for_keyword_search("cat dataset distribution"),
        )

    def test_chinese_segment_stays_whole(self) -> None:
        self.assertEqual(RetrievalRules.tokenize_for_keyword_search("随机投影 matrix"), ["matrix", "随机投影"])

    def test_stopwords_are_dropped_after_stemming(self) -> None:
        tokens = RetrievalRules.tokenize_for_keyword_search("What does this paper use?")

        self.assertEqual(RetrievalRules().extract_query_keywords(tokens), ["use"])


if __name__ == "__main__":
    unittest.main()
