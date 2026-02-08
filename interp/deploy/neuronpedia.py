"""
Upload utilities for self-hosted Neuronpedia instances.

Handles:
- Creating model entries
- Uploading feature dashboard data (batch JSONs)
- Uploading dead feature stubs
- Uploading attribution graphs
"""

from __future__ import annotations

import json
from pathlib import Path

import requests


class NeuronpediaUploader:
    """
    Client for uploading data to a self-hosted Neuronpedia instance.

    Usage:
        uploader = NeuronpediaUploader("http://localhost:3000")
        uploader.create_model("split-llama", num_layers=20)
        uploader.upload_features("split-llama", "0-resid-sae", "./dashboard_data")
    """

    def __init__(self, base_url: str = "http://localhost:3000", api_key: str = ""):
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["x-api-key"] = api_key

    def create_model(
        self,
        model_id: str,
        num_layers: int,
        display_name: str = "",
    ) -> dict:
        """Create a new model entry in Neuronpedia."""
        payload = {
            "id": model_id,
            "layers": num_layers,
        }
        if display_name:
            payload["displayName"] = display_name

        resp = requests.post(
            f"{self.base_url}/api/model/new",
            json=payload,
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json()

    def upload_features(
        self,
        model_id: str,
        source_set: str,
        dashboard_dir: str,
    ) -> int:
        """
        Upload feature dashboard data from batch JSON files.

        Args:
            model_id: The model ID in Neuronpedia.
            source_set: Source set name (e.g., "0-resid-sae").
            dashboard_dir: Directory containing batch-*.json files.

        Returns:
            Number of features uploaded.
        """
        dash_path = Path(dashboard_dir)
        batch_files = sorted(dash_path.glob("batch-*.json"))
        total_features = 0

        for batch_file in batch_files:
            with open(batch_file) as f:
                features = json.load(f)

            # Replace NaN with -999 (Neuronpedia convention)
            features_clean = _replace_nan(features)

            payload = {
                "modelId": model_id,
                "layer": source_set,
                "features": features_clean,
            }

            resp = requests.post(
                f"{self.base_url}/api/local/upload-features",
                json=payload,
                headers=self.headers,
            )
            resp.raise_for_status()
            total_features += len(features_clean)

        return total_features

    def upload_dead_features(
        self,
        model_id: str,
        source_set: str,
        dashboard_dir: str,
    ) -> int:
        """Upload dead feature stubs from skipped_indexes.json."""
        skip_file = Path(dashboard_dir) / "skipped_indexes.json"
        if not skip_file.exists():
            return 0

        with open(skip_file) as f:
            skipped = json.load(f)

        if not skipped:
            return 0

        payload = {
            "modelId": model_id,
            "layer": source_set,
            "deadIndexes": skipped,
        }

        resp = requests.post(
            f"{self.base_url}/api/local/upload-dead-features",
            json=payload,
            headers=self.headers,
        )
        resp.raise_for_status()
        return len(skipped)

    def upload_graph(
        self,
        model_id: str,
        graph_json_path: str,
        prompt: str = "",
        target_token: str = "",
    ) -> dict:
        """Upload an attribution graph to Neuronpedia."""
        with open(graph_json_path) as f:
            graph_data = json.load(f)

        payload = {
            "modelId": model_id,
            "prompt": prompt,
            "targetToken": target_token,
            "graph": graph_data,
        }

        resp = requests.post(
            f"{self.base_url}/api/graph/upload",
            json=payload,
            headers=self.headers,
        )
        resp.raise_for_status()
        return resp.json()

    def health_check(self) -> bool:
        """Check if the Neuronpedia instance is reachable."""
        try:
            resp = requests.get(f"{self.base_url}/api/health", timeout=5)
            return resp.status_code == 200
        except requests.ConnectionError:
            return False


def _replace_nan(obj):
    """Recursively replace NaN/Inf values with -999 for JSON compatibility."""
    if isinstance(obj, float):
        if obj != obj or obj == float("inf") or obj == float("-inf"):
            return -999
        return obj
    elif isinstance(obj, dict):
        return {k: _replace_nan(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_replace_nan(v) for v in obj]
    return obj
