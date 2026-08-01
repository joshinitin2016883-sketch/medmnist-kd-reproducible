"""Evaluate a trained MedMNIST classifier on its held-out test split.

Reads a checkpoint written by train.py, runs the official test split, and emits
per-class precision/recall/F1/support, macro- and weighted-F1, accuracy,
a confusion matrix (stdout + PNG), and per-class AUC where computable.

Every number written to metrics.json comes from this run. Metrics that cannot
be computed are omitted and noted in the "warnings" array -- never filled with
a placeholder.

Example
-------
python evaluate.py --checkpoint checkpoints/resnet_dermamnist.pt --batch-size 64
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings as warnings_module
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# ImageNet statistics -- these MUST match what training used, or every metric
# below is quietly wrong. Defined once here; train.py imports from this module.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
DEFAULT_IMG_SIZE = 224


def to_rgb(img):
    """Promote 1-channel MedMNIST images to 3 channels for ImageNet backbones.

    Module-level (not a lambda) so the transform stays picklable, which is what
    DataLoader workers need on Windows' spawn start method.
    """
    return img.convert("RGB")


def build_eval_transform(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    """Deterministic preprocessing. No augmentation -- this is the test path."""
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(to_rgb),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_model(model_name: str, num_classes: int, pretrained: bool = False) -> nn.Module:
    """Construct an architecture and swap in a fresh `num_classes` head.

    Shared by train.py (pretrained=True) and evaluate.py (pretrained=False), so
    the head surgery is written exactly once. Evaluation defaults to no
    pretrained download because every parameter is about to be overwritten by
    the checkpoint -- fetching ImageNet weights first would just waste bandwidth.
    """
    if model_name == "resnet":
        weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet50(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "densenet":
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.densenet121(weights=weights)
        model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    elif model_name == "efficientnet":
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(
            f"unknown model {model_name!r}; expected resnet, densenet or efficientnet"
        )
    return model


def load_split(dataset_flag, split, data_root, transform, medmnist_size=None):
    """Return (dataset, info) for one official MedMNIST split.

    MedMNIST ships fixed train/val/test splits, so there is no resplitting here
    and no chance of leakage between them -- both scripts use the same official
    partitions the published benchmarks use.
    """
    try:
        import medmnist
        from medmnist import INFO
    except ImportError:
        raise SystemExit(
            "medmnist is not installed. Run:\n"
            "    python -m pip install medmnist==3.0.2"
        )

    if dataset_flag not in INFO:
        raise SystemExit(
            f"unknown dataset {dataset_flag!r}. Available: {', '.join(sorted(INFO))}"
        )

    info = INFO[dataset_flag]
    DataClass = getattr(medmnist, info["python_class"])

    root = Path(data_root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)

    kwargs = dict(split=split, transform=transform, download=True, root=str(root))
    if medmnist_size is not None:
        kwargs["size"] = medmnist_size

    try:
        dataset = DataClass(**kwargs)
    except TypeError as exc:
        if medmnist_size is not None:
            raise SystemExit(
                f"--medmnist-size needs medmnist>=3.0 (native high-res variants). "
                f"Underlying error: {exc}"
            )
        raise

    return dataset, info


@torch.no_grad()
def predict(model, loader, device, tta="none"):
    """Run one pass over `loader`. Returns (probs [N,C], labels [N]).

    tta="flips" averages softmax over the image, its horizontal flip, its
    vertical flip, and both. Dermoscopy has no canonical orientation -- a lesion
    rotated 180 degrees is the same lesion -- so these transforms are genuinely
    label-preserving here. That is not true of every medical modality: flipping
    a chest X-ray horizontally invents dextrocardia.

    Averaging is done on probabilities rather than logits so each view
    contributes on a common, normalised scale.
    """
    model.eval()
    probs_chunks, label_chunks = [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)

        if tta == "flips":
            views = [
                images,
                torch.flip(images, dims=[3]),      # horizontal
                torch.flip(images, dims=[2]),      # vertical
                torch.flip(images, dims=[2, 3]),   # both
            ]
        else:
            views = [images]

        acc = None
        for view in views:
            p = torch.softmax(model(view).float(), dim=1)
            acc = p if acc is None else acc + p
        probs = acc / len(views)

        probs_chunks.append(probs.cpu().numpy())
        # MedMNIST yields labels shaped (B, 1); flatten to (B,).
        label_chunks.append(labels.view(-1).long().numpy())

    return np.vstack(probs_chunks), np.concatenate(label_chunks)


def compute_auc(probs, labels, num_classes, label_names, warnings):
    """Per-class one-vs-rest AUC, plus macro/weighted aggregates.

    A class with no positive or no negative examples in the test split has an
    undefined ROC curve. Those classes are omitted and recorded in `warnings`
    rather than emitted as NaN.
    """
    from sklearn.exceptions import UndefinedMetricWarning
    from sklearn.metrics import roc_auc_score

    auc_block: dict[str, object] = {}

    if num_classes == 2:
        try:
            auc_block["binary"] = float(roc_auc_score(labels, probs[:, 1]))
        except ValueError as exc:
            warnings.append(f"binary AUC not computable: {exc}")
        return auc_block

    per_class: dict[str, float] = {}
    for idx in range(num_classes):
        positives = int((labels == idx).sum())
        if positives == 0 or positives == len(labels):
            warnings.append(
                f"AUC omitted for class '{label_names[idx]}' "
                f"(support={positives} of {len(labels)}; ROC undefined)"
            )
            continue
        binary_truth = (labels == idx).astype(int)
        per_class[label_names[idx]] = float(roc_auc_score(binary_truth, probs[:, idx]))

    if per_class:
        auc_block["per_class_ovr"] = per_class

    # sklearn *warns* rather than raises when a class has no positives, and
    # returns NaN from the macro average. A NaN in metrics.json would be both
    # an invented number and invalid JSON, so screen for it explicitly.
    for average in ("macro", "weighted"):
        try:
            with warnings_module.catch_warnings():
                warnings_module.simplefilter("ignore", UndefinedMetricWarning)
                score = roc_auc_score(
                    labels, probs, multi_class="ovr", average=average,
                    labels=list(range(num_classes)),
                )
        except ValueError as exc:
            warnings.append(f"{average} OvR AUC not computable: {exc}")
            continue

        if not np.isfinite(score):
            warnings.append(
                f"{average} OvR AUC omitted: undefined for at least one class "
                f"(sklearn returned {score})"
            )
            continue
        auc_block[f"{average}_ovr"] = float(score)

    return auc_block


def print_confusion_matrix(cm, label_names):
    """Text confusion matrix. Rows are true classes, columns predicted.

    Columns are indexed rather than named. MedMNIST class names run to 47
    characters ("actinic keratoses and intraepithelial carcinoma") and several
    share a prefix -- truncating them made 'melanoma' and 'melanocytic nevi'
    both render as 'mela', which is worse than useless in a confusion matrix.
    """
    n = len(label_names)
    cell = max(len(str(int(cm.max()))), 4) + 1
    name_w = max(len(x) for x in label_names)

    print("\nConfusion matrix (rows = true, cols = predicted)")
    print(" " * (name_w + 5) + "".join(f"{i:>{cell}}" for i in range(n)))
    for i, name in enumerate(label_names):
        row = "".join(f"{int(v):>{cell}}" for v in cm[i])
        print(f"{name:>{name_w}}  {i} |{row}")


def save_confusion_png(cm, label_names, out_path, title):
    """Row-normalized heatmap annotated with raw counts, rendered with Pillow.

    Row normalization matters here: with one class at ~67% of the test split, a
    raw-count heatmap is one dark square and six invisible ones. Normalizing by
    row turns it into a per-class recall view, while the annotations keep the
    true counts visible.

    Pillow rather than matplotlib deliberately. On this machine Windows
    Application Control blocks matplotlib's `_backend_agg` DLL, and pyplot
    imports it no matter which backend you select -- so every matplotlib
    backend, including the vector ones, fails at import. Pillow's `_imaging`
    is not blocked, and drawing rectangles and text is all this plot needs.
    """
    from PIL import Image, ImageDraw, ImageFont

    row_sums = cm.sum(axis=1, keepdims=True)
    normalized = np.divide(
        cm, row_sums, out=np.zeros_like(cm, dtype=float), where=row_sums != 0
    )

    n = len(label_names)
    cell, pad, top = 84, 16, 76
    left = 54
    legend_line = 20
    width = left + n * cell + pad * 2
    height = top + n * cell + pad * 2 + legend_line * (n + 1) + 20

    try:
        f_title = ImageFont.load_default(size=17)
        f_cell = ImageFont.load_default(size=13)
        f_small = ImageFont.load_default(size=12)
    except TypeError:  # Pillow < 10.1 has no size argument
        f_title = f_cell = f_small = ImageFont.load_default()

    img = Image.new("RGB", (width, height), (255, 255, 255))
    d = ImageDraw.Draw(img)

    def centre(text, x, y, font, fill):
        bbox = d.textbbox((0, 0), text, font=font)
        d.text((x - (bbox[2] - bbox[0]) / 2, y - (bbox[3] - bbox[1]) / 2),
               text, font=font, fill=fill)

    d.text((pad, pad), title, font=f_title, fill=(0, 0, 0))
    d.text((pad, pad + 26), "rows = true, cols = predicted; "
           "cell = count over fraction of true class",
           font=f_small, fill=(90, 90, 90))

    x0, y0 = pad + left, top
    for j in range(n):
        centre(str(j), x0 + j * cell + cell / 2, y0 - 14, f_cell, (0, 0, 0))
    for i in range(n):
        centre(str(i), x0 - 16, y0 + i * cell + cell / 2, f_cell, (0, 0, 0))

    for i in range(n):
        for j in range(n):
            v = float(normalized[i, j])
            # white -> deep blue, linear in the row-normalized value
            fill = (
                int(255 - v * (255 - 8)),
                int(255 - v * (255 - 48)),
                int(255 - v * (255 - 107)),
            )
            box = [x0 + j * cell, y0 + i * cell,
                   x0 + (j + 1) * cell, y0 + (i + 1) * cell]
            d.rectangle(box, fill=fill, outline=(210, 210, 210))

            ink = (255, 255, 255) if v > 0.5 else (20, 20, 20)
            cx, cy = box[0] + cell / 2, box[1] + cell / 2
            centre(str(int(cm[i, j])), cx, cy - 9, f_cell, ink)
            centre(f"{v:.2f}", cx, cy + 9, f_small, ink)

    ly = y0 + n * cell + pad + 6
    d.text((pad, ly), "classes", font=f_small, fill=(0, 0, 0))
    for i, name in enumerate(label_names):
        d.text((pad, ly + legend_line * (i + 1)), f"{i} = {name}",
               font=f_small, fill=(60, 60, 60))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Evaluate a trained MedMNIST classifier on its test split.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--checkpoint", required=True, help="path to .pt written by train.py")
    p.add_argument("--data-root", default="./data", help="MedMNIST download/cache dir")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument(
        "--model", default=None, choices=["resnet", "densenet", "efficientnet"],
        help="override architecture (default: read from checkpoint)",
    )
    p.add_argument(
        "--dataset", default=None,
        help="override MedMNIST flag, e.g. dermamnist (default: read from checkpoint)",
    )
    p.add_argument("--img-size", type=int, default=None, help="override input resolution")
    p.add_argument(
        "--medmnist-size", type=int, default=None, choices=[28, 64, 128, 224],
        help="native MedMNIST+ variant; omit for the original 28px source",
    )
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument(
        "--split", default="test", choices=["test", "val", "train"],
        help="which official split to evaluate; use val to fit thresholds, "
             "never test",
    )
    p.add_argument(
        "--tta", default="none", choices=["none", "flips"],
        help="test-time augmentation: average over h/v flips (4 views)",
    )
    p.add_argument("--out", default="metrics.json")
    p.add_argument("--cm-png", default="confusion_matrix.png")
    p.add_argument(
        "--save-probs", default=None,
        help="optional .npz of raw probabilities + labels, for TTA / threshold "
             "tuning / ensembling without retraining",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv=None):
    from sklearn.metrics import (
        accuracy_score,
        balanced_accuracy_score,
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
    )

    args = parse_args(argv)
    warnings: list[str] = []

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    ckpt_path = Path(args.checkpoint).expanduser().resolve()
    if not ckpt_path.is_file():
        raise SystemExit(f"checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # A checkpoint written by train.py carries its own identity. A bare
    # state_dict does not, so fall back to CLI flags and say so loudly.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        meta = checkpoint
        state_dict = checkpoint["state_dict"]
    else:
        meta = {}
        state_dict = checkpoint
        warnings.append(
            "checkpoint has no metadata (bare state_dict); architecture and "
            "dataset taken from CLI flags -- verify they are correct"
        )

    model_name = args.model or meta.get("model_name")
    dataset_flag = args.dataset or meta.get("dataset_flag")
    img_size = args.img_size or meta.get("img_size") or DEFAULT_IMG_SIZE
    medmnist_size = args.medmnist_size or meta.get("medmnist_size")

    if not model_name:
        raise SystemExit("no architecture in checkpoint; pass --model")
    if not dataset_flag:
        raise SystemExit("no dataset in checkpoint; pass --dataset")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch reports no CUDA device")

    print(f"checkpoint : {ckpt_path}")
    print(f"model      : {model_name}")
    print(f"dataset    : {dataset_flag}"
          + (f" (native {medmnist_size}px)" if medmnist_size else "")
          + f"   split: {args.split}"
          + (f"   TTA: {args.tta}" if args.tta != "none" else ""))
    print(f"input size : {img_size}")
    print(f"device     : {device}")

    transform = build_eval_transform(img_size)
    dataset, info = load_split(
        dataset_flag, args.split, args.data_root, transform, medmnist_size
    )

    label_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    num_classes = len(label_names)

    ckpt_classes = meta.get("num_classes")
    if ckpt_classes is not None and ckpt_classes != num_classes:
        raise SystemExit(
            f"checkpoint has {ckpt_classes} output classes but {dataset_flag} has "
            f"{num_classes}; wrong checkpoint/dataset pairing"
        )

    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(model_name, num_classes)
    # Tolerate DataParallel-prefixed keys without silently dropping mismatches.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.removeprefix("module."): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=True)
    model.to(device)

    started = time.perf_counter()
    probs, labels = predict(model, loader, device, tta=args.tta)
    elapsed = time.perf_counter() - started
    preds = probs.argmax(axis=1)

    accuracy = float(accuracy_score(labels, preds))
    balanced_acc = float(balanced_accuracy_score(labels, preds))
    macro_f1 = float(
        precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)[2]
    )
    weighted_f1 = float(
        precision_recall_fscore_support(labels, preds, average="weighted", zero_division=0)[2]
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, labels=list(range(num_classes)), zero_division=0
    )

    print("\n" + classification_report(
        labels, preds, labels=list(range(num_classes)),
        target_names=label_names, zero_division=0, digits=4,
    ))
    print(f"accuracy          : {accuracy:.4f}")
    print(f"balanced accuracy : {balanced_acc:.4f}")
    print(f"macro-F1          : {macro_f1:.4f}   <- the metric that matters here")
    print(f"weighted-F1       : {weighted_f1:.4f}")

    cm = confusion_matrix(labels, preds, labels=list(range(num_classes)))
    print_confusion_matrix(cm, label_names)

    auc_block = compute_auc(probs, labels, num_classes, label_names, warnings)
    if auc_block:
        print("\nAUC")
        for key, value in auc_block.items():
            if isinstance(value, dict):
                for name, score in value.items():
                    print(f"  {name:>10} : {score:.4f}")
            else:
                print(f"  {key:>10} : {value:.4f}")

    # Rendering is best-effort and must never cost us the metrics. An earlier
    # version wrote metrics.json *after* the plot; when Application Control
    # blocked matplotlib's DLL the whole run died and a completed evaluation
    # was thrown away. Failures are recorded as warnings and carried into the
    # JSON below.
    cm_png = Path(args.cm_png).expanduser().resolve()
    try:
        save_confusion_png(
            cm, label_names, cm_png,
            f"{model_name} / {dataset_flag} ({args.split}, n={len(labels)})",
        )
        print(f"\nconfusion matrix PNG -> {cm_png}")
    except Exception as exc:
        cm_png = None
        warnings.append(
            f"confusion matrix PNG not written: {type(exc).__name__}: {exc}"
        )
        print(f"\nconfusion matrix PNG -> SKIPPED ({type(exc).__name__})")

    metrics = {
        "run": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "checkpoint": str(ckpt_path),
            "model_name": model_name,
            "dataset_flag": dataset_flag,
            "medmnist_size": medmnist_size,
            "img_size": img_size,
            "batch_size": args.batch_size,
            "device": str(device),
            "seed": args.seed,
            "split": args.split,
            "tta": args.tta,
            "n_test_samples": int(len(labels)),
            "inference_seconds": round(elapsed, 3),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "train_run_name": meta.get("run_name"),
            "train_seed": meta.get("seed"),
        },
        "overall": {
            "accuracy": accuracy,
            "balanced_accuracy": balanced_acc,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
        },
        "per_class": [
            {
                "class_index": i,
                "class_name": label_names[i],
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(num_classes)
        ],
        "confusion_matrix": {
            "labels": label_names,
            "counts": cm.astype(int).tolist(),
            "png": str(cm_png) if cm_png else None,
        },
        "warnings": warnings,
    }
    if auc_block:
        metrics["auc"] = auc_block

    out_path = Path(args.out).expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # allow_nan=False turns any NaN/Inf that slipped through into a hard error
    # instead of writing a non-standard JSON literal that most parsers reject.
    try:
        payload = json.dumps(metrics, indent=2, allow_nan=False)
    except ValueError as exc:
        raise SystemExit(f"refusing to write non-finite metric to {out_path}: {exc}")
    out_path.write_text(payload, encoding="utf-8")
    print(f"metrics JSON         -> {out_path}")

    if args.save_probs:
        probs_path = Path(args.save_probs).expanduser().resolve()
        probs_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            probs_path, probs=probs, labels=labels, label_names=np.array(label_names)
        )
        print(f"raw probabilities    -> {probs_path}")

    if warnings:
        print("\nwarnings:")
        for w in warnings:
            print(f"  - {w}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
