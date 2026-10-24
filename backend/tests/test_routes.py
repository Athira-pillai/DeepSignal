import unittest

from backend.main import app


class HistoryRouteTests(unittest.TestCase):
    def test_saved_history_routes_precede_dynamic_symbol_route(self):
        route_paths = [getattr(route, "path", None) for route in app.routes]

        dynamic_history_index = route_paths.index("/api/history/{symbol}")
        for saved_history_path in (
            "/api/history/predictions",
            "/api/history/experiments",
        ):
            with self.subTest(path=saved_history_path):
                self.assertLess(route_paths.index(saved_history_path), dynamic_history_index)


if __name__ == "__main__":
    unittest.main()
