"""Train a MedMNIST classifier and write a checkpoint evaluate.py can read.

Extracted from biomed_kd_final.ipynb. With default flags this reproduces the
notebook's training recipe parameter for parameter:

    Adam(lr=1e-4), batch 64, 224px, max 50 epochs, early stopping on val loss
    (patience 5, best weights restored), AMP autocast + GradScaler,
    augmentation on DermaMNIST only, FocalLoss(gamma=1.5, label_smoothing=0.1)
    with clamped balanced class weights for DermaMNIST, plain unweighted
    CrossEntropyLoss for PneumoniaMNIST and BreastMNIST.

Two things the notebook lacked are added because nothing downstream works
without them: the model is saved, and the run is seeded. The notebook set no
seed, so its published numbers are not reproducible by anyone -- treat the
first seeded run here as the baseline, not as a reproduction of those numbers.

Example
-------
python train.py --model resnet --dataset dermamnist --out checkpoints/resnet_dermamnist.pt
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# Shared with evaluate.py on purpose: if the eval-time transform or the head
# surgery lived in two files they would eventually drift, and every reported
# metric would be silently wrong.
from evaluate import (
    DEFAULT_IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_eval_transform,
    build_model,
    load_split,
    to_rgb,
)

# Datasets the notebook augmented. Pneumonia and Breast were trained on the
# plain eval transform; only Derma got flips/rotation/jitter.
AUGMENTED_BY_DEFAULT = {"dermamnist"}


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed every RNG that affects a run: init, shuffling, augmentation.

    cudnn.benchmark stays on by default because the notebook had it on and it
    is a genuine speedup. It picks convolution algorithms by timing them, which
    makes results slightly nondeterministic. --deterministic trades that speed
    for bit-exact reruns.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True


def build_train_transform(img_size: int, augment: bool) -> transforms.Compose:
    """Training-time preprocessing.

    When augment=False this is byte-identical to the eval transform, which is
    what the notebook used for Pneumonia and Breast.
    """
    if not augment:
        return build_eval_transform(img_size)

    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.Lambda(to_rgb),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def dataset_labels(dataset) -> np.ndarray:
    """Read integer labels without decoding a single image.

    The notebook did `[dataset[i][1] for i in range(len(dataset))]`, which runs
    the full transform pipeline -- decode, resize to 224, augment, normalize --
    on every training image just to read the label off the end. MedMNIST keeps
    labels in a plain array; same values, none of the work.
    """
    labels = getattr(dataset, "labels", None)
    if labels is not None:
        return np.asarray(labels).reshape(-1)
    return np.array([int(dataset[i][1]) for i in range(len(dataset))])


def compute_class_weights(
    labels: np.ndarray,
    num_classes: int,
    device: torch.device,
    power: float = 0.5,
    clamp_min: float = 0.1,
    clamp_max: float = 2.0,
) -> torch.Tensor:
    """Balanced inverse-frequency weights, damped exactly as the notebook did.

    On DermaMNIST proportions, `balanced` puts ~58x more weight on 'df' than on
    'nv'. The notebook damped that three ways -- sqrt, mean-normalise, clamp to
    [0.1, 2.0] -- but only the sqrt actually does anything: it takes the spread
    to sqrt(58) ~= 7.6x, mean-normalise is a pure rescale, and no class ever
    reaches the 2.0 ceiling, so the clamp is inert. The rarest classes are
    therefore under-weighted ~7.6x relative to true inverse frequency.

    Whether that damping helps or hurts macro-F1 is a Phase 2 experiment (vary
    `power`), not something to change silently here.
    """
    from sklearn.utils.class_weight import compute_class_weight

    present = np.unique(labels)
    weights = compute_class_weight("balanced", classes=present, y=labels)

    full = np.ones(num_classes, dtype=np.float64)
    full[present] = weights
    full = full ** power

    tensor = torch.tensor(full, dtype=torch.float, device=device)
    tensor = tensor / tensor.mean()
    return torch.clamp(tensor, min=clamp_min, max=clamp_max)


class FocalLoss(nn.Module):
    """Focal loss with optional per-class alpha weighting.

    label_smoothing is a constructor argument rather than a hardcoded 0.1 (as
    in the notebook) so that focal loss and label smoothing can be varied
    independently. The default reproduces the notebook.
    """

    def __init__(self, gamma: float = 1.5, alpha=None, label_smoothing: float = 0.1):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.label_smoothing = label_smoothing

    def forward(self, logits, targets):
        ce = F.cross_entropy(
            logits, targets, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = (1 - pt) ** self.gamma * ce

        if self.alpha is not None:
            loss = self.alpha.gather(0, targets) * loss

        return loss.mean()


class EarlyStopping:
    """Stop when the monitored metric stops improving; restore best weights.

    mode="min" (the default) reproduces the notebook exactly: monitor val loss,
    lower is better. mode="max" monitors a score such as val macro-F1.

    The distinction matters here. Measured on three seeds, the epoch with the
    best val macro-F1 was consistently *rejected* in favour of the epoch with
    the best val loss -- giving up roughly 0.019 val macro-F1 each time. Val
    loss on DermaMNIST is dominated by `nv` at 67% of the split, so minimising
    it optimises something other than the target metric.

    Snapshots live on CPU so a long run does not hold a second copy of the
    model in VRAM -- on a 6 GB card a second ResNet-50 alongside the training
    graph is enough to OOM.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, mode: str = "min"):
        if mode not in ("min", "max"):
            raise ValueError(f"mode must be 'min' or 'max', got {mode!r}")
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.best_epoch = None
        self.counter = 0
        self.stop = False
        self._best_state = None

    def _is_better(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, model: nn.Module, epoch: int) -> bool:
        improved = self._is_better(score)
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            self._best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True
        return improved

    @property
    def best_loss(self):
        """Backwards-compatible alias for the monitored best value."""
        return self.best_score

    def restore(self, model: nn.Module) -> None:
        if self._best_state is not None:
            model.load_state_dict(self._best_state)

    @property
    def best_state(self):
        return self._best_state


def build_criterion(dataset_flag, loss_choice, class_weights, gamma, label_smoothing):
    """Pick the loss the notebook used for this dataset.

    'auto' reproduces the notebook exactly: focal + weighted alpha on Derma,
    plain unweighted CE elsewhere. That inconsistency is inherited, not
    endorsed -- PneumoniaMNIST is imbalanced too and gets no correction.
    """
    if loss_choice == "auto":
        use_focal = dataset_flag in AUGMENTED_BY_DEFAULT
    else:
        use_focal = loss_choice == "focal"

    if use_focal:
        return FocalLoss(gamma=gamma, alpha=class_weights, label_smoothing=label_smoothing)
    return nn.CrossEntropyLoss()


@torch.no_grad()
def validate(model, loader, criterion, device, num_classes):
    """Return (mean val loss, macro-F1).

    macro-F1 is reported only -- early stopping still watches val loss, exactly
    as the notebook did. It is here because val loss can improve while the rare
    classes get worse, and you want to see that happening.
    """
    from sklearn.metrics import f1_score

    model.eval()
    total, preds, truths = 0.0, [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.view(-1).long().to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(images)
            loss = criterion(outputs, labels)

        total += loss.item()
        preds.append(outputs.float().argmax(1).cpu().numpy())
        truths.append(labels.cpu().numpy())

    macro_f1 = f1_score(
        np.concatenate(truths), np.concatenate(preds),
        labels=list(range(num_classes)), average="macro", zero_division=0,
    )
    return total / max(len(loader), 1), float(macro_f1)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Train a MedMNIST classifier (notebook recipe by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default="resnet",
                   choices=["resnet", "densenet", "efficientnet"])
    p.add_argument("--dataset", default="dermamnist", help="MedMNIST flag")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", default=None,
                   help="checkpoint path (default: checkpoints/<model>_<dataset>.pt)")
    p.add_argument("--history-out", default=None,
                   help="per-epoch losses as JSON (default: alongside checkpoint)")

    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    p.add_argument("--medmnist-size", type=int, default=None, choices=[28, 64, 128, 224],
                   help="native MedMNIST+ variant; omit for the original 28px source")
    p.add_argument("--patience", type=int, default=5)
    p.add_argument(
        "--early-stop-metric", default="val_loss",
        choices=["val_loss", "val_macro_f1"],
        help="quantity early stopping monitors; val_loss reproduces the notebook",
    )

    p.add_argument("--loss", default="auto", choices=["auto", "focal", "ce"],
                   help="auto = focal for dermamnist, plain CE otherwise (notebook)")
    p.add_argument("--focal-gamma", type=float, default=1.5)
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="applied inside focal loss; 0.0 isolates focal on its own")
    p.add_argument("--augment", default="auto", choices=["auto", "on", "off"],
                   help="auto = on for dermamnist only (notebook)")

    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="disable cudnn.benchmark for bit-exact reruns (slower)")
    p.add_argument("--device", default=None, choices=["cuda", "cpu"])
    p.add_argument("--num-workers", type=int, default=0, help="0 is safest on Windows")
    p.add_argument("--run-name", default=None, help="label recorded in the checkpoint")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed, args.deterministic)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but torch reports no CUDA device")
    if device.type == "cpu":
        print("WARNING: training on CPU. This will take hours, not minutes.")

    augment = (
        args.dataset in AUGMENTED_BY_DEFAULT if args.augment == "auto"
        else args.augment == "on"
    )

    train_tf = build_train_transform(args.img_size, augment)
    eval_tf = build_eval_transform(args.img_size)

    train_ds, info = load_split(
        args.dataset, "train", args.data_root, train_tf, args.medmnist_size
    )
    val_ds, _ = load_split(
        args.dataset, "val", args.data_root, eval_tf, args.medmnist_size
    )

    label_names = [info["label"][str(i)] for i in range(len(info["label"]))]
    num_classes = len(label_names)

    train_labels = dataset_labels(train_ds)
    class_weights = compute_class_weights(train_labels, num_classes, device)
    criterion = build_criterion(
        args.dataset, args.loss, class_weights, args.focal_gamma, args.label_smoothing
    )

    run_name = args.run_name or f"{args.model}_{args.dataset}_seed{args.seed}"
    out_path = Path(args.out or f"checkpoints/{args.model}_{args.dataset}.pt").resolve()
    history_path = Path(
        args.history_out or out_path.with_name(out_path.stem + "_history.json")
    ).resolve()

    counts = np.bincount(train_labels, minlength=num_classes)
    print(f"run        : {run_name}")
    print(f"model      : {args.model}   dataset: {args.dataset}   device: {device}")
    print(f"train/val  : {len(train_ds)} / {len(val_ds)}   classes: {num_classes}")
    print(f"augment    : {augment}   loss: {type(criterion).__name__}   seed: {args.seed}")
    print(f"early stop : {args.early_stop_metric} (patience {args.patience})")
    print("class      : " + "  ".join(f"{n}={c}" for n, c in zip(label_names, counts)))
    if isinstance(criterion, FocalLoss):
        print("weights    : " + "  ".join(
            f"{n}={w:.3f}" for n, w in zip(label_names, class_weights.tolist())
        ))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=(device.type == "cuda"),
    )

    model = build_model(args.model, num_classes, pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # Fresh scaler per run. The notebook shared one global GradScaler across all
    # nine trainings, carrying loss-scale state between unrelated models.
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    monitor_mode = "min" if args.early_stop_metric == "val_loss" else "max"
    stopper = EarlyStopping(patience=args.patience, mode=monitor_mode)

    history = []
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0

        for images, labels in tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}"):
            images = images.to(device, non_blocking=True)
            labels = labels.view(-1).long().to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda")):
                loss = criterion(model(images), labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item()

        train_loss = running / max(len(train_loader), 1)
        val_loss, val_macro_f1 = validate(
            model, val_loader, criterion, device, num_classes
        )

        monitored = val_loss if args.early_stop_metric == "val_loss" else val_macro_f1
        improved = stopper.step(monitored, model, epoch)
        marker = "  *best" if improved else ""
        print(
            f"epoch {epoch:>2}/{args.epochs} | train {train_loss:.4f} | "
            f"val {val_loss:.4f} | val macro-F1 {val_macro_f1:.4f}{marker}"
        )
        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_macro_f1": val_macro_f1,
            "best": improved,
        })

        if stopper.stop:
            print(f"early stopping: no improvement for {args.patience} epochs")
            break

    elapsed = time.perf_counter() - started
    stopper.restore(model)
    print(
        f"\nbest val loss {stopper.best_loss:.4f} at epoch {stopper.best_epoch} "
        f"| {len(history)} epochs in {elapsed / 60:.1f} min"
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": stopper.best_state or model.state_dict(),
            "model_name": args.model,
            "dataset_flag": args.dataset,
            "num_classes": num_classes,
            "label_names": label_names,
            "img_size": args.img_size,
            "medmnist_size": args.medmnist_size,
            "seed": args.seed,
            "run_name": run_name,
            "epochs_trained": len(history),
            "best_epoch": stopper.best_epoch,
            "early_stop_metric": args.early_stop_metric,
            "best_monitored_value": stopper.best_score,
            "train_seconds": round(elapsed, 1),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "torch_version": torch.__version__,
            "python_version": platform.python_version(),
            "hyperparams": {
                "optimizer": "Adam",
                "lr": args.lr,
                "batch_size": args.batch_size,
                "max_epochs": args.epochs,
                "patience": args.patience,
                "loss": type(criterion).__name__,
                "focal_gamma": args.focal_gamma if isinstance(criterion, FocalLoss) else None,
                "label_smoothing": args.label_smoothing if isinstance(criterion, FocalLoss) else None,
                "class_weights": class_weights.tolist() if isinstance(criterion, FocalLoss) else None,
                "augment": augment,
                "deterministic": args.deterministic,
            },
        },
        out_path,
    )
    print(f"checkpoint -> {out_path}")

    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(
        json.dumps(
            {"run_name": run_name, "train_seconds": round(elapsed, 1), "epochs": history},
            indent=2, allow_nan=False,
        ),
        encoding="utf-8",
    )
    print(f"history    -> {history_path}")
    print(f"\nnext:\n  python evaluate.py --checkpoint {out_path} --batch-size {args.batch_size}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
