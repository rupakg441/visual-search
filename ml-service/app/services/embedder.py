from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from PIL import Image

from app.config import Settings
from app.models.attributes import ZeroShotAttributeScorer
from app.models.base import ImageEncoder, l2_normalize
from app.models.clip_encoder import OpenCLIPEncoder
from app.models.dinov2_encoder import DINOv2Encoder
from app.preprocessing.background import BackgroundRemover
from app.preprocessing.detector import ObjectDetector
from app.preprocessing.transforms import build_tta_views
from app.preprocessing.validator import validate_and_decode_image


@dataclass
class EmbeddingResult:
    embedding: list[float]
    dims: int
    secondary_embedding: list[float] | None = None
    secondary_dims: int | None = None
    predicted_attributes: dict[str, dict[str, float | str]] | None = None


class EmbeddingPipeline:
    """Shared preprocessing + embedding pipeline for both catalog indexing and query images."""

    def __init__(
        self,
        settings: Settings,
        primary_encoder: ImageEncoder,
        dinov2_encoder: DINOv2Encoder | None = None,
        detector: ObjectDetector | None = None,
        bg_remover: BackgroundRemover | None = None,
        attribute_scorer: ZeroShotAttributeScorer | None = None,
    ) -> None:
        self.settings = settings
        self.primary_encoder = primary_encoder
        self.dinov2_encoder = dinov2_encoder
        self.detector = detector
        self.bg_remover = bg_remover
        self.attribute_scorer = attribute_scorer

    @classmethod
    def from_settings(cls, settings: Settings) -> "EmbeddingPipeline":
        detector = (
            ObjectDetector(settings.yolo_model, settings.yolo_conf)
            if settings.enable_object_detection
            else None
        )
        if detector:
            detector.load()

        bg_remover = BackgroundRemover() if settings.enable_bg_removal else None
        if bg_remover:
            bg_remover.load()

        attribute_scorer: ZeroShotAttributeScorer | None = None
        if settings.embedding_backend == "dinov2":
            primary: ImageEncoder = DINOv2Encoder(settings.dinov2_model, settings.device)
            primary.load()
        else:
            model_name = (
                settings.siglip_model
                if settings.embedding_backend == "siglip"
                else settings.openclip_model
            )
            pretrained = (
                settings.siglip_pretrained
                if settings.embedding_backend == "siglip"
                else settings.openclip_pretrained
            )
            clip_enc = OpenCLIPEncoder(
                model_name=model_name,
                pretrained=pretrained,
                device=settings.device,
                enable_onnx=settings.enable_onnx,
                onnx_dir=settings.onnx_model_dir,
                finetuned_weights_path=settings.finetuned_weights_path,
            )
            clip_enc.load()
            primary = clip_enc
            if settings.enable_zero_shot_attributes:
                attribute_scorer = ZeroShotAttributeScorer(clip_enc)

        dinov2: DINOv2Encoder | None = None
        if settings.enable_dinov2 and settings.embedding_backend != "dinov2":
            dinov2 = DINOv2Encoder(settings.dinov2_model, settings.device)
            dinov2.load()

        return cls(
            settings=settings,
            primary_encoder=primary,
            dinov2_encoder=dinov2,
            detector=detector,
            bg_remover=bg_remover,
            attribute_scorer=attribute_scorer,
        )

    def preprocess_image(
        self,
        raw_bytes: bytes,
        content_type: str | None = None,
        skip_bg_removal: bool = False,
        skip_detection: bool = False,
    ) -> Image.Image:
        img = validate_and_decode_image(
            raw_bytes=raw_bytes,
            content_type=content_type,
            max_bytes=self.settings.max_image_bytes,
        )
        if self.detector and not skip_detection:
            img = self.detector.crop_main_object(img)
        if self.bg_remover and not skip_bg_removal:
            img = self.bg_remover.remove_to_white(img)
        return img

    def _encode_with_tta(
        self,
        encoder: ImageEncoder,
        images: list[Image.Image],
        use_tta: bool,
    ) -> np.ndarray:
        if not use_tta:
            return encoder.encode_images(images)

        results = []
        for img in images:
            views = build_tta_views(img)
            view_embs = encoder.encode_images(views)
            mean_emb = np.mean(view_embs, axis=0)
            results.append(l2_normalize(mean_emb))
        return np.stack(results, axis=0)

    def embed_batch(
        self,
        items: list[tuple[bytes, str | None]],
        use_tta: bool | None = None,
        skip_bg_removal: bool = False,
        skip_detection: bool = False,
    ) -> list[EmbeddingResult]:
        tta_flag = self.settings.enable_tta if use_tta is None else use_tta
        processed = [
            self.preprocess_image(
                raw,
                ctype,
                skip_bg_removal=skip_bg_removal,
                skip_detection=skip_detection,
            )
            for raw, ctype in items
        ]

        primary_embs = self._encode_with_tta(self.primary_encoder, processed, tta_flag)
        secondary_embs = (
            self._encode_with_tta(self.dinov2_encoder, processed, tta_flag)
            if self.dinov2_encoder is not None
            else None
        )

        results: list[EmbeddingResult] = []
        for idx in range(len(processed)):
            prim = primary_embs[idx]
            sec = secondary_embs[idx] if secondary_embs is not None else None
            attrs = (
                self.attribute_scorer.predict(prim)
                if self.attribute_scorer is not None
                else None
            )
            results.append(
                EmbeddingResult(
                    embedding=prim.tolist(),
                    dims=int(prim.shape[0]),
                    secondary_embedding=sec.tolist() if sec is not None else None,
                    secondary_dims=int(sec.shape[0]) if sec is not None else None,
                    predicted_attributes=attrs,
                )
            )
        return results
