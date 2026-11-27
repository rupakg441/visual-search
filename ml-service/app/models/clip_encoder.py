from __future__ import annotations

import ctypes
import gc
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING
import numpy as np
from PIL import Image

from app.models.base import l2_normalize

if TYPE_CHECKING:
    import torch

logger = logging.getLogger(__name__)

# Standard OpenCLIP RGB normalization constants (used by pure-ONNX fast path)
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def trim_memory() -> None:
    """Force Python GC and ask glibc to return freed heap pages to the OS."""
    gc.collect()
    if os.name == "posix":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except OSError:
            pass


def resolve_device(requested: str = "auto") -> str:
    import torch

    cleaned = requested.strip().lower()
    if not cleaned or cleaned == "auto" or cleaned.startswith("auto"):
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    return requested.strip()


def _pil_clip_preprocess(images: list[Image.Image], size: int = 224) -> np.ndarray:
    """Lightweight PIL+NumPy CLIP resize/center-crop/normalize matching torchvision OpenCLIP transforms."""
    batch = []
    for img in images:
        w, h = img.size
        if w < h:
            new_w, new_h = size, max(size, int(round(h * size / w)))
        else:
            new_w, new_h = max(size, int(round(w * size / h))), size
        resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)
        left = (new_w - size) // 2
        top = (new_h - size) // 2
        cropped = resized.crop((left, top, left + size, top + size))
        arr = np.asarray(cropped, dtype=np.float32) / 255.0
        arr = (arr - CLIP_MEAN) / CLIP_STD
        batch.append(np.transpose(arr, (2, 0, 1)))
    return np.stack(batch, axis=0).astype(np.float32)


class OpenCLIPEncoder:
    """OpenCLIP / SigLIP vision encoder with low-memory loading and optional ONNX Runtime acceleration."""

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
        self.requested_device = device
        self.device = "cpu"
        self.enable_onnx = enable_onnx
        self.onnx_dir = Path(onnx_dir)
        self.finetuned_weights_path = finetuned_weights_path

        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self.ort_session = None
        self.image_size: int = 224
        self.dims: int = 512

    def _onnx_path(self) -> Path:
        safe_name = f"{self.model_name}_{self.pretrained}".replace("/", "_")
        return self.onnx_dir / f"{safe_name}_visual.onnx"

    def _open_ort_session(self, onnx_path: Path):
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.enable_cpu_mem_arena = False
        opts.enable_mem_pattern = False
        opts.intra_op_num_threads = int(os.getenv("TORCH_NUM_THREADS", "2"))
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self.device == "cuda"
            else ["CPUExecutionProvider"]
        )
        return ort.InferenceSession(str(onnx_path), sess_options=opts, providers=providers)

    def load(self) -> None:
        # Fast low-RAM path: if visual ONNX already exists on disk and no fine-tuned PyTorch checkpoint is given,
        # load ONLY the ONNX visual graph without importing torch or instantiating PyTorch weights at all.
        onnx_path = self._onnx_path()
        if self.enable_onnx and onnx_path.exists() and not self.finetuned_weights_path:
            self.ort_session = self._open_ort_session(onnx_path)
            in_shape = self.ort_session.get_inputs()[0].shape
            out_shape = self.ort_session.get_outputs()[0].shape
            if isinstance(in_shape[-1], int):
                self.image_size = int(in_shape[-1])
            if isinstance(out_shape[-1], int):
                self.dims = int(out_shape[-1])
            logger.info(
                "Loaded prebuilt visual ONNX %s | size=%d | dims=%d (0 PyTorch weights in RAM)",
                onnx_path,
                self.image_size,
                self.dims,
            )
            return

        import open_clip
        import torch

        self.device = resolve_device(self.requested_device)
        torch.set_num_threads(int(os.getenv("TORCH_NUM_THREADS", "2")))

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name,
            pretrained=self.pretrained,
            device=self.device,
        )
        if self.finetuned_weights_path:
            ckpt = torch.load(
                self.finetuned_weights_path,
                map_location=self.device,
                weights_only=False,
            )
            state_dict = ckpt.get("model_state_dict", ckpt)
            model.load_state_dict(state_dict, strict=False, assign=True)
            del ckpt, state_dict
            logger.info("Loaded fine-tuned weights from %s", self.finetuned_weights_path)

        model.eval()
        self.model = model
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)

        # Probe output dimension from visual module
        with torch.inference_mode():
            h = w = 224
            if hasattr(model.visual, "image_size"):
                img_sz = model.visual.image_size
                h, w = img_sz if isinstance(img_sz, tuple) else (img_sz, img_sz)
            self.image_size = int(h)
            dummy = torch.zeros(1, 3, h, w, device=self.device)
            feats = model.encode_image(dummy)
            self.dims = int(feats.shape[-1])

        if self.enable_onnx:
            self._init_onnx(dummy)

        trim_memory()
        logger.info(
            "Loaded OpenCLIP %s (%s) on %s | dims=%d | onnx=%s",
            self.model_name,
            self.pretrained,
            self.device,
            self.dims,
            self.ort_session is not None,
        )

    def release_unused_components(self, keep_text: bool = False) -> None:
        """Drop unused text transformer weights (and PyTorch visual weights if ONNX is active) to slash RAM."""
        if not keep_text and self.model is not None:
            for attr in (
                "transformer",
                "token_embedding",
                "positional_embedding",
                "ln_final",
                "text_projection",
                "logit_scale",
                "text",
            ):
                if hasattr(self.model, attr):
                    try:
                        delattr(self.model, attr)
                    except AttributeError:
                        setattr(self.model, attr, None)
            self.tokenizer = None

        if self.ort_session is not None and not keep_text and self.model is not None:
            self.model = None
            self.preprocess = None

        trim_memory()

    def _init_onnx(self, dummy_input: "torch.Tensor") -> None:
        import torch

        try:
            self.onnx_dir.mkdir(parents=True, exist_ok=True)
            onnx_path = self._onnx_path()
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

            self.ort_session = self._open_ort_session(onnx_path)
        except Exception as exc:  # pragma: no cover - optional runtime export
            logger.warning("ONNX init failed (%s); falling back to PyTorch.", exc)
            self.ort_session = None

    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        if self.ort_session is not None:
            if self.preprocess is not None:
                import torch

                tensors = torch.stack([self.preprocess(img) for img in images]).numpy()
            else:
                tensors = _pil_clip_preprocess(images, size=self.image_size)
            input_name = self.ort_session.get_inputs()[0].name
            outputs = self.ort_session.run(None, {input_name: tensors})
            embeddings = np.asarray(outputs[0], dtype=np.float32)
            return l2_normalize(embeddings)

        if self.model is None or self.preprocess is None:
            raise RuntimeError("OpenCLIPEncoder is not loaded.")

        import torch

        tensors = torch.stack([self.preprocess(img) for img in images])
        with torch.inference_mode():
            feats = self.model.encode_image(tensors.to(self.device))
            embeddings = feats.detach().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)

    def encode_texts(self, prompts: list[str]) -> np.ndarray:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("OpenCLIPEncoder text tower is not loaded.")

        import torch

        tokens = self.tokenizer(prompts).to(self.device)
        with torch.inference_mode():
            feats = self.model.encode_text(tokens)
            embeddings = feats.detach().cpu().numpy().astype(np.float32)
        return l2_normalize(embeddings)
