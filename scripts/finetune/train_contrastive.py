from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import random
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


class ContrastiveJewelryDataset(Dataset):
    """Pairs two views of the same productId (cross-angle or click-through pair) for contrastive learning."""

    def __init__(self, pairs: list[tuple[Path, Path]], preprocess) -> None:
        self.pairs = pairs
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        anchor_path, pos_path = self.pairs[idx]
        anchor = self.preprocess(Image.open(anchor_path).convert("RGB"))
        positive = self.preprocess(Image.open(pos_path).convert("RGB"))
        return anchor, positive


def build_pairs_from_catalog(catalog_path: Path, image_dir: Path) -> list[tuple[Path, Path]]:
    items = json.loads(catalog_path.read_text(encoding="utf-8"))
    by_product: dict[str, list[Path]] = defaultdict(list)
    for item in items:
        p = image_dir / item["localPath"]
        if p.exists():
            by_product[item["productId"]].append(p)

    pairs: list[tuple[Path, Path]] = []
    for paths in by_product.values():
        if len(paths) >= 2:
            for i in range(len(paths) - 1):
                pairs.append((paths[i], paths[i + 1]))
    return pairs


def infonce_loss(anchor_emb: torch.Tensor, pos_emb: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    anchor_emb = F.normalize(anchor_emb, dim=-1)
    pos_emb = F.normalize(pos_emb, dim=-1)
    logits = (anchor_emb @ pos_emb.T) / temperature
    labels = torch.arange(logits.size(0), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune OpenCLIP visual encoder with InfoNCE/triplet loss")
    parser.add_argument("--catalog", type=Path, default=Path("../../data/catalog.json"))
    parser.add_argument("--image-dir", type=Path, default=Path("../../data/images"))
    parser.add_argument("--model-name", type=str, default="ViT-B-32")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--output-version", type=str, default="openclip-vit-b32-ft-v1")
    parser.add_argument("--out-dir", type=Path, default=Path("../../models/finetuned"))
    args = parser.parse_args()

    import open_clip

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained, device=device
    )

    pairs = build_pairs_from_catalog(args.catalog, args.image_dir)
    if not pairs:
        raise SystemExit("No multi-image pairs found in catalog.json.")
    random.shuffle(pairs)

    dataset = ContrastiveJewelryDataset(pairs, preprocess)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=len(pairs) >= args.batch_size)

    # Fine-tune only visual projection & last transformer block to avoid overfitting on small catalogs
    for param in model.parameters():
        param.requires_grad = False
    for param in model.visual.parameters():
        param.requires_grad = True

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=1e-4)
    triplet = nn.TripletMarginLoss(margin=0.2)

    model.train()
    for epoch in range(1, args.epochs + 1):
        running_loss = 0.0
        for anchor, positive in loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            emb_a = F.normalize(model.encode_image(anchor), dim=-1)
            emb_p = F.normalize(model.encode_image(positive), dim=-1)

            loss = infonce_loss(emb_a, emb_p)
            if emb_a.size(0) > 1:
                neg = torch.roll(emb_p, shifts=1, dims=0)
                loss = loss + 0.25 * triplet(emb_a, emb_p, neg)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        print(f"Epoch {epoch}/{args.epochs} | avg_loss={running_loss / max(1, len(loader)):.4f}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.out_dir / f"{args.output_version}.pt"
    torch.save(
        {
            "model_version": args.output_version,
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "model_state_dict": model.state_dict(),
        },
        ckpt_path,
    )
    print(f"Saved fine-tuned checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
