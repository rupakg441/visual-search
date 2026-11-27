from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from app.config import Settings, get_settings
from app.preprocessing.validator import ImageValidationError
from app.services.embedder import EmbeddingPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

_pipeline: EmbeddingPipeline | None = None


def get_pipeline() -> EmbeddingPipeline:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="ML pipeline is not initialized.")
    return _pipeline


def create_app(
    settings: Settings | None = None,
    pipeline_override: EmbeddingPipeline | None = None,
) -> FastAPI:
    cfg = settings or get_settings()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        global _pipeline
        if pipeline_override is not None:
            _pipeline = pipeline_override
        else:
            _pipeline = EmbeddingPipeline.from_settings(cfg)
        yield
        _pipeline = None

    app = FastAPI(
        title="Jewelry Visual Search ML Service",
        version="0.1.0",
        lifespan=lifespan,
    )

    class EmbedResponse(BaseModel):
        embedding: list[float]
        dims: int
        modelVersion: str
        secondaryEmbedding: list[float] | None = None
        secondaryDims: int | None = None
        predictedAttributes: dict[str, dict[str, float | str]] | None = None

    class BatchEmbedItem(EmbedResponse):
        filename: str

    class BatchEmbedResponse(BaseModel):
        embeddings: list[BatchEmbedItem]
        modelVersion: str

    @app.get("/health")
    def health() -> dict:
        pipe = get_pipeline()
        return {
            "status": "ok",
            "modelVersion": cfg.model_version,
            "backend": cfg.embedding_backend,
            "dims": pipe.primary_encoder.dims,
            "dinov2Enabled": pipe.dinov2_encoder is not None,
            "dinov2Weight": cfg.dinov2_weight if pipe.dinov2_encoder is not None else 0.0,
            "objectDetectionEnabled": pipe.detector is not None,
            "bgRemovalEnabled": pipe.bg_remover is not None,
            "ttaEnabled": cfg.enable_tta,
            "onnxEnabled": cfg.enable_onnx,
        }

    @app.post("/embed", response_model=EmbedResponse)
    async def embed_single(
        file: UploadFile = File(...),
        tta: bool | None = Query(default=None),
        skip_bg_removal: bool = Query(default=False),
        skip_detection: bool = Query(default=False),
    ) -> EmbedResponse:
        pipe = get_pipeline()
        raw_bytes = await file.read(cfg.max_image_bytes + 1)
        try:
            res = pipe.embed_batch(
                [(raw_bytes, file.content_type)],
                use_tta=tta,
                skip_bg_removal=skip_bg_removal,
                skip_detection=skip_detection,
            )[0]
        except ImageValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return EmbedResponse(
            embedding=res.embedding,
            dims=res.dims,
            modelVersion=cfg.model_version,
            secondaryEmbedding=res.secondary_embedding,
            secondaryDims=res.secondary_dims,
            predictedAttributes=res.predicted_attributes,
        )

    @app.post("/embed/batch", response_model=BatchEmbedResponse)
    async def embed_batch(
        files: list[UploadFile] = File(...),
        tta: bool | None = Query(default=None),
        skip_bg_removal: bool = Query(default=False),
        skip_detection: bool = Query(default=False),
    ) -> BatchEmbedResponse:
        if not files:
            raise HTTPException(status_code=400, detail="At least one image file is required.")
        if len(files) > 32:
            raise HTTPException(status_code=400, detail="A batch may contain at most 32 images.")
        pipe = get_pipeline()
        payload = [(await f.read(cfg.max_image_bytes + 1), f.content_type) for f in files]
        try:
            results = pipe.embed_batch(
                payload,
                use_tta=tta,
                skip_bg_removal=skip_bg_removal,
                skip_detection=skip_detection,
            )
        except ImageValidationError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return BatchEmbedResponse(
            modelVersion=cfg.model_version,
            embeddings=[
                BatchEmbedItem(
                    filename=f.filename or f"image-{idx}",
                    embedding=res.embedding,
                    dims=res.dims,
                    modelVersion=cfg.model_version,
                    secondaryEmbedding=res.secondary_embedding,
                    secondaryDims=res.secondary_dims,
                    predictedAttributes=res.predicted_attributes,
                )
                for idx, (f, res) in enumerate(zip(files, results))
            ],
        )

    return app


app = create_app()
