from __future__ import annotations

import io
import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models.base import l2_normalize
from app.services.embedder import EmbeddingPipeline


class FakeEncoder:
    dims = 8

    def load(self) -> None:
        return None

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        vecs = []
        for img in images:
            arr = np.asarray(img.resize((8, 1)), dtype=np.float32).mean(axis=0).mean(axis=-1)
            vecs.append(arr + 1.0)
        return l2_normalize(np.stack(vecs, axis=0))


def _make_png_bytes(color: tuple[int, int, int] = (220, 180, 60)) -> bytes:
    img = Image.new("RGB", (64, 64), color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_health_and_embed_endpoints() -> None:
    cfg = Settings(
        model_version="test-clip-v1",
        enable_object_detection=False,
        enable_bg_removal=False,
        enable_zero_shot_attributes=False,
    )
    pipeline = EmbeddingPipeline(settings=cfg, primary_encoder=FakeEncoder())
    app = create_app(settings=cfg, pipeline_override=pipeline)

    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["status"] == "ok"
        assert health.json()["dims"] == 8

        png = _make_png_bytes()
        resp = client.post(
            "/embed",
            files={"file": ("ring.png", png, "image/png")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["dims"] == 8
        assert body["modelVersion"] == "test-clip-v1"
        norm = np.linalg.norm(np.array(body["embedding"], dtype=np.float32))
        assert np.isclose(norm, 1.0, atol=1e-4)

        batch_resp = client.post(
            "/embed/batch",
            files=[
                ("files", ("a.png", png, "image/png")),
                ("files", ("b.png", _make_png_bytes((100, 120, 200)), "image/png")),
            ],
        )
        assert batch_resp.status_code == 200
        assert len(batch_resp.json()["embeddings"]) == 2


def test_embed_rejects_invalid_and_oversized_files() -> None:
    cfg = Settings(
        max_image_bytes=128,
        enable_object_detection=False,
        enable_bg_removal=False,
        enable_zero_shot_attributes=False,
    )
    pipeline = EmbeddingPipeline(settings=cfg, primary_encoder=FakeEncoder())
    app = create_app(settings=cfg, pipeline_override=pipeline)

    with TestClient(app) as client:
        bad_type = client.post(
            "/embed",
            files={"file": ("notes.txt", b"hello", "text/plain")},
        )
        assert bad_type.status_code == 400

        too_big = client.post(
            "/embed",
            files={"file": ("big.png", _make_png_bytes(), "image/png")},
        )
        assert too_big.status_code == 400

        corrupt = client.post(
            "/embed",
            files={"file": ("bad.png", b"not-a-real-png", "image/png")},
        )
        assert corrupt.status_code == 400
