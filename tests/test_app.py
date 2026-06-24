import unittest

from fastapi.testclient import TestClient

from src.app.main import app


class AppShellTests(unittest.TestCase):
    def test_index_serves_model_selector_without_cache(self):
        with TestClient(app) as client:
            response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn('id="model"', response.text)
        self.assertIn("no-store", response.headers["cache-control"])

    def test_models_endpoint_lists_selectable_models(self):
        with TestClient(app) as client:
            response = client.get("/models")

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            [model["name"] for model in payload["models"]],
            ["cnn", "svm", "phowhisper"],
        )


if __name__ == "__main__":
    unittest.main()
