from __future__ import annotations

import argparse
from pathlib import Path
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description="Export fine-tuned OpenCLIP visual encoder to ONNX")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx-out", type=Path, required=True)
    args = parser.parse_args()

    import open_clip

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt.get("model_name", "ViT-B-32")
    pretrained = ckpt.get("pretrained", "laion2b_s34b_b79k")

    model, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained, device="cpu")
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()

    dummy = torch.zeros(1, 3, 224, 224)
    args.onnx_out.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model.visual,
        dummy,
        str(args.onnx_out),
        input_names=["image"],
        output_names=["embedding"],
        dynamic_axes={"image": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported ONNX visual encoder for {ckpt.get('model_version')} -> {args.onnx_out}")


if __name__ == "__main__":
    main()
