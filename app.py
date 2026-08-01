"""Flask API serving a trained MedMNIST classifier, with Grad-CAM explanations.

Endpoints
---------
GET  /health          model/device status
GET  /classes         class index -> label mapping
POST /predict         multipart image -> class probabilities
POST /explain         multipart image -> Grad-CAM overlay (PNG)

This serves a research model trained on 28x28 benchmark thumbnails. It is not a
diagnostic tool; every response carries a disclaimer field saying so. See
MODEL_CARD.md.

Run
---
    $env:MODEL_PATH = "checkpoints/baseline_resnet_derma_seed42.pt"
    python -m flask --app app run --port 8000
"""

from __future__ import annotations

import io
import os
import time
from pathlib import Path

import numpy as np
import torch
from flask import Flask, jsonify, request, send_file
from PIL import Image

# Shared with training and evaluation. Serving preprocessing that drifts from
# eval preprocessing is the classic train/serve skew bug -- one definition only.
from evaluate import build_eval_transform, build_model

DISCLAIMER = (
    "Research artifact, not a medical device. Trained on 28x28 MedMNIST "
    "benchmark images. Must not be used for diagnosis, screening or triage. "
    "See MODEL_CARD.md."
)

MAX_UPLOAD_BYTES = 8 * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

_state: dict = {}


def load_model():
    """Load the checkpoint once at startup and cache it in module state."""
    ckpt_path = Path(os.environ.get("MODEL_PATH", "checkpoints/baseline_resnet_derma_seed42.pt"))
    if not ckpt_path.is_file():
        raise SystemExit(
            f"checkpoint not found: {ckpt_path.resolve()}\n"
            "Set MODEL_PATH, or train one first:\n"
            "  python train.py --model resnet --dataset dermamnist"
        )

    # CPU by default: serving one image at a time gains nothing from a GPU, and
    # it keeps the container image ~2.5 GB smaller.
    device = torch.device(os.environ.get("DEVICE", "cpu"))
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not (isinstance(ckpt, dict) and "state_dict" in ckpt):
        raise SystemExit(
            f"{ckpt_path} is a bare state_dict with no metadata; cannot infer "
            "architecture or class list. Retrain with train.py."
        )

    labels = ckpt.get("label_names") or [str(i) for i in range(ckpt["num_classes"])]
    model = build_model(ckpt["model_name"], ckpt["num_classes"], pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model.to(device).eval()

    _state.update(
        model=model,
        device=device,
        labels=labels,
        img_size=ckpt.get("img_size", 224),
        transform=build_eval_transform(ckpt.get("img_size", 224)),
        meta={
            "checkpoint": str(ckpt_path),
            "model_name": ckpt["model_name"],
            "dataset_flag": ckpt["dataset_flag"],
            "num_classes": ckpt["num_classes"],
            "train_run_name": ckpt.get("run_name"),
            "train_seed": ckpt.get("seed"),
        },
    )
    return _state


def read_image(req) -> Image.Image:
    """Pull an RGB image out of the request, or raise ValueError."""
    if "image" not in req.files:
        raise ValueError("no file part named 'image' in the request")
    f = req.files["image"]
    if not f.filename:
        raise ValueError("empty filename")
    try:
        return Image.open(io.BytesIO(f.read())).convert("RGB")
    except Exception as exc:
        raise ValueError(f"could not decode image: {exc}")


def predict_probs(img: Image.Image) -> np.ndarray:
    tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])
    with torch.no_grad():
        logits = _state["model"](tensor)
    return torch.softmax(logits.float(), dim=1)[0].cpu().numpy()


def target_layer(model, model_name):
    """Last conv block -- the deepest layer that still has spatial extent."""
    if model_name == "resnet":
        return model.layer4[-1]
    if model_name in ("densenet", "efficientnet"):
        return model.features[-1]
    raise ValueError(f"no Grad-CAM target layer defined for {model_name!r}")


class GradCAM:
    """Grad-CAM with hooks scoped to a `with` block.

    Three fixes over the notebook implementation:
      * `register_full_backward_hook`, not the deprecated
        `register_backward_hook`, which fires unreliably on modules whose
        forward does anything non-trivial;
      * hooks are removed on exit. The notebook registered a new pair on every
        call and never removed them, so repeated use stacked hooks on the same
        model and leaked memory;
      * a zero-activation map no longer divides by zero.
    """

    def __init__(self, model, layer):
        self.model, self.layer = model, layer
        self.activations = self.gradients = None
        self._handles = []

    def __enter__(self):
        self._handles = [
            self.layer.register_forward_hook(
                lambda m, i, o: setattr(self, "activations", o)
            ),
            self.layer.register_full_backward_hook(
                lambda m, gi, go: setattr(self, "gradients", go[0])
            ),
        ]
        return self

    def __exit__(self, *exc):
        for h in self._handles:
            h.remove()
        self._handles = []

    def generate(self, tensor, class_idx: int) -> np.ndarray:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(tensor)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * self.activations).sum(dim=1))[0]

        cam = cam - cam.min()
        peak = cam.max()
        # An all-zero map means no positive evidence reached this layer. Return
        # a flat map rather than propagating NaN through a divide by zero.
        cam = cam / peak if peak > 0 else torch.zeros_like(cam)
        return cam.detach().cpu().numpy()


def jet(values: np.ndarray) -> np.ndarray:
    """Jet colormap, matching the notebook's visual convention.

    Hand-rolled because matplotlib is unavailable on some target machines --
    Windows Application Control blocks its `_backend_agg` DLL, and pyplot
    imports that regardless of the selected backend.
    """
    v = np.clip(values, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4 * v - 3), 0, 1)
    g = np.clip(1.5 - np.abs(4 * v - 2), 0, 1)
    b = np.clip(1.5 - np.abs(4 * v - 1), 0, 1)
    return (np.stack([r, g, b], axis=-1) * 255).astype(np.uint8)


def overlay_cam(img: Image.Image, cam: np.ndarray, alpha: float = 0.5) -> Image.Image:
    """Blend a Grad-CAM heatmap over the resized input image.

    The CAM is 7x7 for a 224px ResNet input and is resized explicitly here. The
    notebook relied on matplotlib stretching it implicitly at draw time, which
    works but hides the upsampling from the reader.
    """
    size = _state["img_size"]
    base = img.resize((size, size), Image.BILINEAR)
    heat = Image.fromarray(jet(cam)).resize((size, size), Image.BILINEAR)
    return Image.blend(base, heat, alpha)


@app.get("/health")
def health():
    ok = "model" in _state
    return jsonify(
        status="ok" if ok else "model not loaded",
        device=str(_state.get("device", "")),
        model=_state.get("meta", {}),
        disclaimer=DISCLAIMER,
    ), (200 if ok else 503)


@app.get("/classes")
def classes():
    return jsonify(
        classes={i: n for i, n in enumerate(_state["labels"])},
        disclaimer=DISCLAIMER,
    )


@app.post("/predict")
def predict():
    try:
        img = read_image(request)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    started = time.perf_counter()
    probs = predict_probs(img)
    order = np.argsort(probs)[::-1]

    return jsonify(
        prediction={
            "class_index": int(order[0]),
            "class_name": _state["labels"][order[0]],
            "probability": float(probs[order[0]]),
        },
        probabilities={_state["labels"][i]: float(probs[i]) for i in order},
        inference_ms=round((time.perf_counter() - started) * 1000, 2),
        # Softmax outputs are uncalibrated; see MODEL_CARD.md limitation 4.
        calibrated=False,
        disclaimer=DISCLAIMER,
    )


@app.post("/explain")
def explain():
    try:
        img = read_image(request)
    except ValueError as exc:
        return jsonify(error=str(exc)), 400

    tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])
    model = _state["model"]

    requested = request.args.get("class_index")
    if requested is not None:
        try:
            idx = int(requested)
        except ValueError:
            return jsonify(error="class_index must be an integer"), 400
        if not 0 <= idx < len(_state["labels"]):
            return jsonify(error=f"class_index out of range 0..{len(_state['labels']) - 1}"), 400
    else:
        with torch.no_grad():
            idx = int(model(tensor).argmax(1).item())

    with GradCAM(model, target_layer(model, _state["meta"]["model_name"])) as cam_gen:
        cam = cam_gen.generate(tensor, idx)

    buf = io.BytesIO()
    overlay_cam(img, cam).save(buf, format="PNG")
    buf.seek(0)

    resp = send_file(buf, mimetype="image/png")
    resp.headers["X-Explained-Class"] = _state["labels"][idx]
    resp.headers["X-Disclaimer"] = "not a diagnostic tool"
    return resp


@app.errorhandler(413)
def too_large(_):
    return jsonify(error=f"upload exceeds {MAX_UPLOAD_BYTES // (1024 * 1024)} MB"), 413


load_model()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
