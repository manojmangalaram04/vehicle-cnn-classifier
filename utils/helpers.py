"""Shared constants and helpers. The class list is the single source of truth
for training, evaluation and the web app: it is written to models/class_names.json
by train.py (from the folders in data/train) and read back everywhere else."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "validation"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MODEL_PATH = MODELS_DIR / "vehicle_classifier.keras"
CLASS_NAMES_PATH = MODELS_DIR / "class_names.json"

IMG_SIZE = 128
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
FALLBACK_CLASSES = ["bicycle", "car", "motorbike", "truck"]

EMOJI = {"car": "🚗", "bicycle": "🚲", "motorbike": "🏍️", "truck": "🚚", "bus": "🚌",
         "auto_rickshaw": "🛺", "train": "🚆", "airplane": "✈️", "ship": "🚢"}


def read_json(path):
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def discover_class_names(directory: Path) -> list[str]:
    return sorted(p.name for p in Path(directory).iterdir() if p.is_dir())


def load_class_names() -> list[str]:
    data = read_json(CLASS_NAMES_PATH)
    if data and data.get("class_names"):
        return list(data["class_names"])
    if TRAIN_DIR.exists():
        names = discover_class_names(TRAIN_DIR)
        if names:
            return names
    return list(FALLBACK_CLASSES)


def pretty(name: str) -> str:
    return name.replace("_", " ").title()


def emoji(name: str) -> str:
    return EMOJI.get(name, "🚘")


def ensure_dirs() -> None:
    for d in (MODELS_DIR, OUTPUTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def load_pil_image(file_or_path) -> Image.Image:
    """Open an image, fix EXIF rotation, return RGB. Raises ValueError if unreadable."""
    try:
        img = Image.open(file_or_path)
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):          # flatten transparency onto white
            img = img.convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
            bg.alpha_composite(img)
            img = bg
        return img.convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unreadable or invalid image: {exc}") from exc


def preprocess_pil_image(img: Image.Image) -> np.ndarray:
    """RGB -> 128x128 -> float32 batch (1,128,128,3), values 0-255.
    Normalisation (1/255) is a layer inside the model, so every code path matches."""
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.expand_dims(np.asarray(img, dtype="float32"), axis=0)


def predict_image(model, img: Image.Image, class_names: list[str]):
    probs = model.predict(preprocess_pil_image(img), verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx] * 100.0), probs
