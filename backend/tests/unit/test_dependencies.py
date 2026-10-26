import unittest
from unittest import mock

import dependencies


class DependenciesUnitTests(unittest.TestCase):
    def test_normalize_service_load_mode_supports_preload_aliases(self) -> None:
        self.assertEqual(dependencies.normalize_service_load_mode("preload"), "preload")
        self.assertEqual(dependencies.normalize_service_load_mode("EAGER"), "preload")
        self.assertEqual(dependencies.normalize_service_load_mode("startup-load"), "preload")

    def test_normalize_service_load_mode_defaults_to_lazy(self) -> None:
        self.assertEqual(dependencies.normalize_service_load_mode("lazy"), "lazy")
        self.assertEqual(dependencies.normalize_service_load_mode("unknown"), "lazy")
        self.assertEqual(dependencies.normalize_service_load_mode(None), "preload" if dependencies.SERVICE_LOAD_MODE == "preload" else "lazy")

    def test_iter_service_getters_exposes_expected_structure(self) -> None:
        getters = dependencies.iter_service_getters()

        self.assertIsInstance(getters, list)
        self.assertGreaterEqual(len(getters), 10)
        self.assertEqual(getters[0][0], "storage_container")
        names = [name for name, _ in getters]
        # 快速问答复用详细模式的会话服务，预热时必须排在 paper_qa_service 之后。
        self.assertEqual(names[-1], "paper_fast_qa_service")
        self.assertLess(names.index("paper_qa_service"), names.index("paper_fast_qa_service"))
        self.assertTrue(all(isinstance(name, str) and callable(getter) for name, getter in getters))

    def test_warm_up_services_lazy_mode_returns_empty_without_touching_getters(self) -> None:
        with mock.patch.object(dependencies, "iter_service_getters", side_effect=AssertionError("should not be called")):
            warmed = dependencies.warm_up_services(load_mode="lazy")

        self.assertEqual(warmed, [])


if __name__ == "__main__":
    unittest.main()
