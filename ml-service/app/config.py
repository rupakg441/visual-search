from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=(".env", "../.env"), extra="ignore")

    model_version: str = "openclip-vit-b32-v1"
    embedding_backend: Literal["openclip", "siglip", "dinov2"] = "openclip"
    openclip_model: str = "ViT-B-32"
    openclip_pretrained: str = "laion2b_s34b_b79k"
    siglip_model: str = "ViT-B-16-SigLIP"
    siglip_pretrained: str = "webli"

    enable_dinov2: bool = False
    dinov2_model: str = "facebook/dinov2-base"
    dinov2_weight: float = 0.35

    device: str = "auto"
    enable_onnx: bool = False
    onnx_model_dir: str = "./models/onnx"

    max_image_bytes: int = 10 * 1024 * 1024
    enable_object_detection: bool = True
    yolo_model: str = "yolov8n.pt"
    yolo_conf: float = 0.25
    enable_bg_removal: bool = True
    enable_tta: bool = False
    enable_zero_shot_attributes: bool = True

    finetuned_weights_path: str | None = None


@lru_cache
def get_settings() -> Settings:
    return Settings()
