from __future__ import annotations

import logging
from pathlib import Path
import numpy as np
from PIL import Image
import torch

from app.models.base import l2_normalize

logger = logging.getLogger(__name__)


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class OpenCLIPEncoder:
    """OpenCLIP / SigLIP vision encoder with optional ONNX Runtime acceleration."""

    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str = "auto",
        enable_onnx: bool = False,
        onnx_dir: str = "./models/onnx",
        finetuned_weights_path: str | None = None,
    ) -> None:
        self.model_name = model_name
        self.pretrained = pretrained
        self.device = resolve_device(device)
        self.enable_onnx = enable_onnx
        self.onnx_dir = Path(onnx_dir)
        self.finetuned_weights_path = finetuned_weights_path

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.ort_session = None
        self.dims: int = 512

    def load(self) -> None:
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        if self.finetuned_weights_path:
            ckpt = torch.load(self.finetuned_weights_path, map_location=self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False)
            logger.info("Loaded fine-tuned weights from %s", self.finetuned_weights_path)

        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Probe output dimension from visual module
        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224, device=self.device)
            if hasattr(model.visual, "image_size"):
                img_sz = model.visual.image_size
                h, w = img_sz if isinstance(img_sz, tuple) else (img_sz, img_sz)
                dummy = torch.zeros(1, 3, h, w, device=self.device)
            feats = model.encode_image(dummy)
            self.dims = int(feats.shape[-1])

        if self.enable_onnx:
            self._init_onnx(dummy)

        logger.info(
            "Loaded OpenCLIP %s (%s) on %s | dims=%d | onnx=%s",
            self.model_name,
            self.pretrained,
            self.device,
            self.dims,
            self.ort_session is not None,
        )

    def _init_onnx(self, dummy_input: torch.Tensor) -> None:
        try:
            import onnxruntime as ort

            self.onnx_dir.mkdir(parents=True, exist_ok=True)
            safe_name = f"{self.model_name}_{self.pretrained}".replace("/", "_")
            onnx_path = self.onnx_dir / f"{safe_name}_visual.onnx"
            if not onnx_path.exists():
                torch.onnx.export(
                    self.model.visual,
                    dummy_input.cpu(),
                    str(onnx_path),
                    input_names=["image"],
                    output_names=["embedding"],
                    dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
                    opset_version=17,
                )
                logger.info("Exported visual encoder to ONNX: %s", onnx_path)

            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if self.device == "cuda"
                else ["CPUExecutionProvider"]
            )
            self.ort_session = ort.InferenceSession(str(onnx_path), providers=providers)
        except Exception as exc:  # pragma: no cover - optional runtime export
            logger.warning("ONNX init failed (%s); falling back to PyTorch.", exc)
            self.ort_session = None

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if self.model is None or self.preprocess is None:
            raise RuntimeError("OpenCLIPEncoder is not loaded.")

        tensors = torch.stack([self.preprocess(img) for img in images])
        if self.ort_session is not None:
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: tensors.numpy()})
            embeddings = np.asarray(outputs[0], dtype=np.float32)
            return l2_normalize(embeddings)

        with torch.no_grad():
            feats = self.model.encode_image(tensors.to(self.device))
            embeddings = feats.detach().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)

    def encode_texts(self, prompts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("OpenCLIPEncoder is not loaded.")
        tokens = self.tokenizer(prompts).to(self.device)
        with torch.no_grad():
            feats = self.model.encode_text(tokens)
            embeddings = feats.detach().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)
