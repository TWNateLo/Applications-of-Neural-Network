from __future__ import annotations

# -----------------------------------------------------------------------------
# HW4 CNN Training
# -----------------------------------------------------------------------------
#   How to run the training:
#   1. Run all training for all 3 experiments:
#   python train.py --raw_dir . --run_all
#   2. Run the deeper160 training and the final training for all-data deeper160:
#   python train.py --raw_dir . --config deeper160 --final_train_all --epochs 100
# -----------------------------------------------------------------------------

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# Fix common Windows Intel OpenMP duplicated DLL issue before torch/cv2 import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
# Keep matplotlib font cache local to the project folder on Windows/school PCs.
os.environ.setdefault("MPLCONFIGDIR", str((Path.cwd() / ".matplotlib_cache").resolve()))

import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

torch.set_num_threads(1)

CLASS_NAMES = ["hao", "jin", "gua"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =============================================================================
# 1. Image preprocessing
# =============================================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def safe_open_image(path: str | Path) -> Image.Image:
    """Open a possibly rotated phone image and return RGB pixels."""
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def letterbox_resize(img: Image.Image, size: int) -> Image.Image:
    """Resize image to size x size without cropping original content.

    This is safer than center crop for hidden full-body images because it keeps
    the whole person/face visible and pads the shorter side using the image mean
    color instead of black bars.
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    scale = min(size / w, size / h)
    new_w = max(1, int(w * scale))
    new_h = max(1, int(h * scale))
    resized = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    arr = np.asarray(resized)
    mean_color = tuple(np.mean(arr.reshape(-1, 3), axis=0).astype(np.uint8).tolist())
    canvas = Image.new("RGB", (size, size), mean_color)
    canvas.paste(resized, ((size - new_w) // 2, (size - new_h) // 2))
    return canvas


def center_crop_view(img: Image.Image, margin_ratio: float = 0.92) -> Image.Image:
    """A mild center crop used only as an optional test-time view."""
    w, h = img.size
    crop_w, crop_h = int(w * margin_ratio), int(h * margin_ratio)
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def upper_body_crop_view(img: Image.Image) -> Image.Image:
    """A portrait-oriented upper-body/face view used only for TTA."""
    w, h = img.size
    if h <= w:
        return center_crop_view(img, 0.92)
    return img.crop((0, 0, w, int(h * 0.72)))


def detect_face_crop(img: Image.Image, margin: float = 1.75) -> Image.Image | None:
    """Optional OpenCV face crop.

    Returns None if OpenCV is unavailable or if no face is detected. The full
    image is always still used, so this function never removes the main input.
    """
    try:
        import cv2
    except Exception:
        return None

    try:
        gray = np.asarray(img.convert("L"))
        cascade_path = str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml")
        detector = cv2.CascadeClassifier(cascade_path)
        faces = detector.detectMultiScale(gray, scaleFactor=1.08, minNeighbors=4, minSize=(24, 24))
    except Exception:
        return None

    if len(faces) == 0:
        return None

    x, y, w, h = max(faces, key=lambda box: box[2] * box[3])
    cx, cy = x + w / 2, y + h / 2
    side = max(w, h) * margin
    left = max(0, int(cx - side / 2))
    top = max(0, int(cy - side / 2))
    right = min(img.width, int(cx + side / 2))
    bottom = min(img.height, int(cy + side / 2))
    if right <= left or bottom <= top:
        return None
    return img.crop((left, top, right, bottom))


def image_to_tensor(img: Image.Image, size: int) -> torch.Tensor:
    """PIL image -> normalized CHW tensor in [-1, 1]."""
    img = letterbox_resize(img, size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def augment_tensor(x: torch.Tensor) -> torch.Tensor:
    """Training-only tensor augmentation.

    This is applied only after the train/val/test split. It never modifies the
    raw image files on disk.
    """
    x = x.clone()

    if torch.rand(()) < 0.50:
        x = torch.flip(x, dims=[2])

    if torch.rand(()) < 0.80:
        # Brightness/contrast jitter in [0, 1] space.
        z = x * 0.5 + 0.5
        brightness = float(torch.empty(()).uniform_(0.75, 1.25))
        contrast = float(torch.empty(()).uniform_(0.75, 1.25))
        z = z * brightness
        mean = z.mean(dim=(1, 2), keepdim=True)
        z = (z - mean) * contrast + mean
        z = z + torch.randn(3, 1, 1) * 0.03
        x = (z.clamp(0, 1) - 0.5) / 0.5

    if torch.rand(()) < 0.25:
        # Random erasing / occlusion simulation.
        _, h, w = x.shape
        area = h * w * float(torch.empty(()).uniform_(0.02, 0.07))
        aspect = float(torch.empty(()).uniform_(0.6, 1.8))
        erase_h = max(1, int(round(math.sqrt(area * aspect))))
        erase_w = max(1, int(round(math.sqrt(area / aspect))))
        if erase_h < h and erase_w < w:
            yy = random.randint(0, h - erase_h)
            xx = random.randint(0, w - erase_w)
            x[:, yy : yy + erase_h, xx : xx + erase_w] = torch.randn(3, erase_h, erase_w) * 0.20

    return x


# =============================================================================
# 2. Dataset collection and split
# =============================================================================

def collect_raw_samples(raw_dir: str | Path) -> List[dict]:
    """Collect images from data/, data_face_only/, train/, val/, test/, or raw class folders.

    Supported layouts:
        raw_dir/data/{hao,jin,gua}/...
        raw_dir/data_face_only/{hao,jin,gua}/...
        raw_dir/{hao,jin,gua}/...
        raw_dir/train/{hao,jin,gua}/...
    """
    raw_dir = Path(raw_dir)
    samples: List[dict] = []

    candidate_roots: List[Tuple[str, Path]] = []
    for sub in ["data", "data_face_only", "train", "val", "test", ""]:
        root = raw_dir / sub if sub else raw_dir
        if all((root / cls).exists() for cls in CLASS_NAMES):
            candidate_roots.append((sub or "raw", root))

    if not candidate_roots:
        raise FileNotFoundError(
            f"Could not find class folders {CLASS_NAMES} under {raw_dir}.\n"
            "Expected data/{hao,jin,gua}, data_face_only/{hao,jin,gua}, "
            "train/{hao,jin,gua}, or directly {hao,jin,gua}."
        )

    for source_name, root in candidate_roots:
        for cls in CLASS_NAMES:
            class_dir = root / cls
            for path in sorted(class_dir.rglob("*")):
                if is_image_file(path):
                    samples.append(
                        {
                            "path": str(path),
                            "class_name": cls,
                            "label": CLASS_NAMES.index(cls),
                            "source": source_name,
                            # Put data/img001.jpg and data_face_only/img001.jpg in the same split.
                            "group": f"{cls}_{path.stem.lower()}",
                        }
                    )

    return samples


def stratified_group_split(
    samples: List[dict],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42,
) -> Dict[str, List[dict]]:
    """Class-balanced group-aware split to reduce data leakage."""
    rng = random.Random(seed)
    split = {"train": [], "val": [], "test": []}

    for cls in CLASS_NAMES:
        groups: Dict[str, List[dict]] = defaultdict(list)
        for sample in samples:
            if sample["class_name"] == cls:
                groups[sample["group"]].append(sample)

        keys = list(groups.keys())
        rng.shuffle(keys)
        n = len(keys)
        if n < 3:
            raise ValueError(f"Class {cls!r} has too few image groups ({n}). Need at least 3.")

        n_train = max(1, round(n * train_ratio))
        n_val = max(1, round(n * val_ratio))
        if n_train + n_val >= n:
            n_train = max(1, n - 2)
            n_val = 1

        for key in keys[:n_train]:
            split["train"].extend(groups[key])
        for key in keys[n_train : n_train + n_val]:
            split["val"].extend(groups[key])
        for key in keys[n_train + n_val :]:
            split["test"].extend(groups[key])

    return split


def save_split_manifest(split: Dict[str, List[dict]], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "class_name", "label", "source", "group", "path"])
        for split_name, items in split.items():
            for s in items:
                writer.writerow([split_name, s["class_name"], s["label"], s["source"], s["group"], s["path"]])


class FaceImageDataset(Dataset):
    """Image dataset with cached resized tensors and training-only augmentation."""

    def __init__(
        self,
        samples: List[dict],
        input_size: int = 128,
        train: bool = False,
        add_face_crop_view: bool = False,
    ):
        self.samples = samples
        self.input_size = input_size
        self.train = train
        self.add_face_crop_view = add_face_crop_view
        self.tensors: List[torch.Tensor] = []
        self.labels: List[int] = []
        self.paths: List[str] = []
        self._build_cache()

    def _build_cache(self) -> None:
        for s in self.samples:
            img = safe_open_image(s["path"])
            self.tensors.append(image_to_tensor(img, self.input_size))
            self.labels.append(int(s["label"]))
            self.paths.append(s["path"])

            # Additional face crop training view. It does not replace the full image.
            if self.train and self.add_face_crop_view:
                face = detect_face_crop(img)
                if face is not None:
                    self.tensors.append(image_to_tensor(face, self.input_size))
                    self.labels.append(int(s["label"]))
                    self.paths.append(s["path"] + "#face_crop")

    def __len__(self) -> int:
        return len(self.tensors)

    def __getitem__(self, idx: int):
        x = self.tensors[idx]
        y = self.labels[idx]
        if self.train:
            x = augment_tensor(x)
        return x, y, self.paths[idx]


def make_loaders(
    split: Dict[str, List[dict]],
    input_size: int,
    batch_size: int,
    add_face_crop_view: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    print(f"Building cached datasets at input_size={input_size} ...", flush=True)
    train_ds = FaceImageDataset(split["train"], input_size=input_size, train=True, add_face_crop_view=add_face_crop_view)
    val_ds = FaceImageDataset(split["val"], input_size=input_size, train=False)
    test_ds = FaceImageDataset(split["test"], input_size=input_size, train=False)
    print(
        f"Dataset tensors: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}",
        flush=True,
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


# =============================================================================
# 3. CNN models
# =============================================================================

class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1):
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class BaselineCNN(nn.Module):
    """Assignment-style baseline. Expected input: 3 x 64 x 64."""

    def __init__(self, num_classes: int = 3, dropout: float = 0.50):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CompactFaceCNN(nn.Module):
    """Improved compact CNN for small mixed face/full-body datasets."""

    def __init__(self, num_classes: int = 3, base: int = 24, dropout: float = 0.35):
        super().__init__()
        b = base
        self.features = nn.Sequential(
            ConvBNAct(3, b),
            ConvBNAct(b, b),
            nn.MaxPool2d(2),
            ConvBNAct(b, b * 2),
            ConvBNAct(b * 2, b * 2),
            nn.MaxPool2d(2),
            ConvBNAct(b * 2, b * 4),
            ConvBNAct(b * 4, b * 4),
            nn.MaxPool2d(2),
            ConvBNAct(b * 4, b * 6),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(b * 6, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.features(x))


class DeeperFaceCNN(nn.Module):
    """Deeper CNN option for 160x160 final training."""

    def __init__(self, num_classes: int = 3, base: int = 24, dropout: float = 0.40):
        super().__init__()
        b = base
        self.features = nn.Sequential(
            ConvBNAct(3, b),
            ConvBNAct(b, b),
            nn.MaxPool2d(2),
            ConvBNAct(b, b * 2),
            ConvBNAct(b * 2, b * 2),
            nn.MaxPool2d(2),
            ConvBNAct(b * 2, b * 4),
            ConvBNAct(b * 4, b * 4),
            nn.MaxPool2d(2),
            ConvBNAct(b * 4, b * 6),
            ConvBNAct(b * 6, b * 6),
            nn.MaxPool2d(2),
            ConvBNAct(b * 6, b * 8),
            nn.AdaptiveAvgPool2d(1),
        )
        self.cls = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(b * 8, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cls(self.features(x))


def create_model(arch: str = "compact", num_classes: int = 3, base: int = 24, dropout: float = 0.35) -> nn.Module:
    arch = arch.lower()
    if arch == "baseline":
        return BaselineCNN(num_classes=num_classes, dropout=dropout)
    if arch == "compact":
        return CompactFaceCNN(num_classes=num_classes, base=base, dropout=dropout)
    if arch == "deeper":
        return DeeperFaceCNN(num_classes=num_classes, base=base, dropout=dropout)
    raise ValueError(f"Unknown arch={arch!r}; choose baseline, compact, or deeper.")


# =============================================================================
# 4. Training
# =============================================================================

def get_configs() -> Dict[str, dict]:
    return {
        "baseline64": {
            "arch": "baseline",
            "input_size": 64,
            "base": 16,
            "dropout": 0.50,
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 0.0,
            "batch_size": 16,
            "epochs": 35,
            "label_smoothing": 0.0,
            "add_face_crop_view": False,
        },
        "compact128": {
            "arch": "compact",
            "input_size": 128,
            "base": 24,
            "dropout": 0.35,
            "optimizer": "adamw",
            "lr": 8e-4,
            "weight_decay": 1e-4,
            "batch_size": 16,
            "epochs": 60,
            "label_smoothing": 0.05,
            "add_face_crop_view": True,
        },
        "deeper160": {
            "arch": "deeper",
            "input_size": 160,
            "base": 24,
            "dropout": 0.40,
            "optimizer": "adamw",
            "lr": 6e-4,
            "weight_decay": 2e-4,
            "batch_size": 12,
            "epochs": 80,
            "label_smoothing": 0.05,
            "add_face_crop_view": True,
        },
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def make_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "momentum":
        return torch.optim.SGD(
            model.parameters(), lr=cfg["lr"], momentum=0.9, nesterov=True, weight_decay=cfg["weight_decay"]
        )
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    raise ValueError(f"Unsupported optimizer: {cfg['optimizer']}")


def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    cm = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)

    context = torch.enable_grad() if train else torch.no_grad()
    with context:
        for x, y, _paths in loader:
            x = x.to(device)
            y = y.to(device)
            if train:
                optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()

            pred = logits.argmax(dim=1)
            bs = y.size(0)
            total_loss += loss.item() * bs
            total_correct += (pred == y).sum().item()
            total_count += bs
            for true_label, pred_label in zip(y.detach().cpu(), pred.detach().cpu()):
                cm[true_label, pred_label] += 1

    return total_loss / total_count, total_correct / total_count, cm


def save_history_csv(history: List[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)


def save_plots(history: List[dict], out_dir: Path, name: str) -> None:
    """Save loss/accuracy plots using matplotlib."""
    if not history:
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    # Use matplotlib for the required loss/accuracy visualizations.
    # The Agg backend saves PNG files without opening a GUI window.
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    epochs = [int(row["epoch"]) for row in history]

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(row["train_loss"]) for row in history], label="train_loss")
    plt.plot(epochs, [float(row["val_loss"]) for row in history], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{name} loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_loss.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 4))
    plt.plot(epochs, [float(row["train_acc"]) for row in history], label="train_acc")
    plt.plot(epochs, [float(row["val_acc"]) for row in history], label="val_acc")
    plt.plot(epochs, [float(row["test_acc"]) for row in history], label="test_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{name} accuracy")
    plt.ylim(0.0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{name}_accuracy.png", dpi=180)
    plt.close()

def _clean_display_path(path_value) -> str:
    """Remove internal suffixes such as '#face_crop' before opening images."""
    return str(path_value).split("#", 1)[0]


def save_prediction_examples(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_png: Path,
    out_csv: Path | None = None,
    max_per_class: int = 2,
) -> None:
    """Save HW4-required prediction examples for a model/run.

    HW4 asks for at least two test images from each class and requires the
    examples to clearly show true labels and predicted labels. This function is
    called after each experiment, using that experiment's best checkpoint and
    its held-out test split.

    The selected examples are random every time this function is called. This
    intentionally does not use args.seed, so repeated training/test runs can
    produce different example images for the report while keeping model training
    itself reproducible.
    """
    model.eval()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    if out_csv is not None:
        out_csv.parent.mkdir(parents=True, exist_ok=True)

    all_rows: List[dict] = []
    rows_by_class: Dict[int, List[dict]] = {idx: [] for idx in range(len(CLASS_NAMES))}

    with torch.no_grad():
        for x, y, paths in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1).detach().cpu()
            preds = probs.argmax(dim=1)

            for true_idx_tensor, pred_idx_tensor, prob_tensor, path_value in zip(y, preds, probs, paths):
                true_idx = int(true_idx_tensor)
                pred_idx = int(pred_idx_tensor)
                row = {
                    "path": _clean_display_path(path_value),
                    "true_label": CLASS_NAMES[true_idx],
                    "pred_label": CLASS_NAMES[pred_idx],
                    "prob_hao": float(prob_tensor[0]),
                    "prob_jin": float(prob_tensor[1]),
                    "prob_gua": float(prob_tensor[2]),
                }
                all_rows.append(row)
                rows_by_class[true_idx].append(row)

    if not all_rows:
        print(f"No prediction examples were available for {out_png}", flush=True)
        return

    # Use SystemRandom so example selection changes each time, even though the
    # training seed is fixed for reproducibility.
    rng = random.SystemRandom()
    chosen: List[dict] = []
    missing: List[str] = []
    for class_idx in range(len(CLASS_NAMES)):
        candidates = rows_by_class[class_idx]
        if len(candidates) < max_per_class:
            missing.append(CLASS_NAMES[class_idx])
            chosen.extend(candidates)
        else:
            chosen.extend(rng.sample(candidates, max_per_class))

    # Keep the final figure easy to read: grouped by true class, with random
    # selections inside each class group.
    chosen.sort(key=lambda r: (CLASS_NAMES.index(r["true_label"]), rng.random()))

    if out_csv is not None:
        with out_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["path", "true_label", "pred_label", "prob_hao", "prob_jin", "prob_gua"],
            )
            writer.writeheader()
            writer.writerows(chosen)

    if missing:
        print(
            f"Warning: fewer than {max_per_class} test examples found for classes: {missing}. "
            f"Saved available examples only.",
            flush=True,
        )

    tile_w = 360
    tile_h = 430
    text_h = 60
    img_area_h = tile_h - text_h
    cols = max(1, max_per_class)
    rows = math.ceil(len(chosen) / cols)
    canvas = Image.new("RGB", (cols * tile_w, rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 22)
    except Exception:
        font = ImageFont.load_default()

    for i, row in enumerate(chosen):
        col = i % cols
        row_idx = i // cols
        x0 = col * tile_w
        y0 = row_idx * tile_h

        img = ImageOps.exif_transpose(Image.open(row["path"])).convert("RGB")
        img.thumbnail((tile_w - 16, img_area_h - 16), Image.Resampling.BICUBIC)
        img_x = x0 + (tile_w - img.width) // 2
        img_y = y0 + text_h + (img_area_h - img.height) // 2
        canvas.paste(img, (img_x, img_y))

        correct = row["true_label"] == row["pred_label"]
        color = (0, 128, 0) if correct else (220, 0, 0)
        label_text = f"True: {row['true_label']}\nPred: {row['pred_label']}"
        draw.multiline_text((x0 + 10, y0 + 8), label_text, fill=color, font=font, spacing=3)

    canvas.save(out_png)
    print(f"Saved random prediction examples: {out_png}", flush=True)
    if out_csv is not None:
        print(f"Saved random prediction example CSV: {out_csv}", flush=True)

def train_one_config(name: str, cfg: dict, split: dict, args) -> Tuple[float, Path]:
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"\nPreparing config={name} on device={device}", flush=True)

    train_loader, val_loader, test_loader = make_loaders(
        split,
        input_size=cfg["input_size"],
        batch_size=cfg["batch_size"],
        add_face_crop_view=cfg["add_face_crop_view"],
        num_workers=args.num_workers,
    )

    model = create_model(cfg["arch"], num_classes=len(CLASS_NAMES), base=cfg["base"], dropout=cfg["dropout"]).to(device)
    optimizer = make_optimizer(model, cfg)
    epochs = args.epochs or cfg["epochs"]
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    best_val_acc = -1.0
    best_val_loss = 999.0
    best_state = None
    history: List[dict] = []
    patience_counter = 0

    print(f"=== Training {name} for {epochs} epochs ===", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_loss, val_acc, _ = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        test_loss, test_acc, _ = run_epoch(model, test_loader, criterion, optimizer, device, train=False)
        scheduler.step()

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        if val_acc > best_val_acc or (abs(val_acc - best_val_acc) < 1e-12 and val_loss < best_val_loss):
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_acc={train_acc:.3f} val_acc={val_acc:.3f} test_acc={test_acc:.3f} "
            f"train_loss={train_loss:.3f} val_loss={val_loss:.3f}",
            flush=True,
        )

        if patience_counter >= args.patience and epoch >= args.min_epochs:
            print(f"Early stopping after {epoch} epochs.", flush=True)
            break

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_history_csv(history, out_dir / f"history_{name}.csv")
    save_plots(history, out_dir, name)

    ckpt_path = Path(args.model_dir) / f"{name}_best.pth"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "arch": cfg["arch"],
            "class_names": CLASS_NAMES,
            "input_size": cfg["input_size"],
            "base": cfg["base"],
            "dropout": cfg["dropout"],
            "config": cfg,
            "best_val_acc": best_val_acc,
            "best_val_loss": best_val_loss,
            "history": history,
        },
        ckpt_path,
    )
    print(f"Saved best checkpoint: {ckpt_path} (best_val_acc={best_val_acc:.3f})", flush=True)

    # HW4-required prediction examples: use the held-out test split and the
    # best validation checkpoint for this specific experiment/config.
    if not args.no_prediction_examples and best_state is not None:
        model.load_state_dict(best_state)
        save_prediction_examples(
            model=model,
            loader=test_loader,
            device=device,
            out_png=out_dir / f"prediction_examples_{name}.png",
            out_csv=out_dir / f"prediction_examples_{name}.csv",
            max_per_class=args.examples_per_class,
        )

    return best_val_acc, ckpt_path


def train_final_on_all(samples: List[dict], cfg: dict, args, example_samples: List[dict] | None = None) -> Path:
    """Retrain selected config on all images and save models/best_model.pth."""
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    epochs = args.epochs or cfg["epochs"]
    print(f"\nPreparing final training dataset at input_size={cfg['input_size']} ...", flush=True)
    dataset = FaceImageDataset(
        samples,
        input_size=cfg["input_size"],
        train=True,
        add_face_crop_view=cfg["add_face_crop_view"],
    )
    loader = DataLoader(dataset, batch_size=cfg["batch_size"], shuffle=True, num_workers=args.num_workers)
    print(f"Final training tensors: {len(dataset)}", flush=True)

    model = create_model(cfg["arch"], num_classes=len(CLASS_NAMES), base=cfg["base"], dropout=cfg["dropout"]).to(device)
    optimizer = make_optimizer(model, cfg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs))
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])

    print(f"=== Final training on ALL available images for {epochs} epochs ===", flush=True)
    for epoch in range(1, epochs + 1):
        train_loss, train_acc, _ = run_epoch(model, loader, criterion, optimizer, device, train=True)
        scheduler.step()
        print(f"Final epoch {epoch:03d}/{epochs} | train_acc={train_acc:.3f} train_loss={train_loss:.3f}", flush=True)

    best_path = Path(args.model_dir) / "best_model.pth"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "arch": cfg["arch"],
            "class_names": CLASS_NAMES,
            "input_size": cfg["input_size"],
            "base": cfg["base"],
            "dropout": cfg["dropout"],
            "config": cfg,
            "trained_on_all_available_data": True,
        },
        best_path,
    )
    print(f"Saved final hidden-test checkpoint: {best_path}", flush=True)

    # Optional report figure for the final all-data model. These examples are
    # taken from the original split's test images for visualization only.
    # Because --final_train_all trains on all available data, this figure should
    # not be treated as an independent test metric; it is for the HW4 visual
    # prediction-example requirement.
    if not args.no_prediction_examples and example_samples:
        print("Building final-model prediction examples from original test split ...", flush=True)
        example_ds = FaceImageDataset(example_samples, input_size=cfg["input_size"], train=False)
        example_loader = DataLoader(example_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=args.num_workers)
        save_prediction_examples(
            model=model,
            loader=example_loader,
            device=device,
            out_png=Path(args.out_dir) / f"prediction_examples_final_{args.config}.png",
            out_csv=Path(args.out_dir) / f"prediction_examples_final_{args.config}.csv",
            max_per_class=args.examples_per_class,
        )

    return best_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=str, default=".", help="Folder containing data/ and data_face_only/, or class folders.")
    parser.add_argument("--config", type=str, default="compact128", choices=list(get_configs().keys()))
    parser.add_argument("--run_all", action="store_true", help="Run all required experiment configs.")
    parser.add_argument("--final_train_all", action="store_true", help="Retrain selected config on all images and save models/best_model.pth.")
    parser.add_argument("--epochs", type=int, default=None, help="Override config epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=18)
    parser.add_argument("--min_epochs", type=int, default=20)
    parser.add_argument("--out_dir", type=str, default="outputs")
    parser.add_argument("--model_dir", type=str, default="models")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--examples_per_class", type=int, default=2, help="Number of test prediction examples to save per class.")
    parser.add_argument("--no_prediction_examples", action="store_true", help="Disable output of prediction example PNG/CSV files.")
    args = parser.parse_args()

    print("Starting train.py", flush=True)
    print(f"Working directory: {Path.cwd()}", flush=True)
    print(f"raw_dir: {Path(args.raw_dir).resolve()}", flush=True)

    set_seed(args.seed)
    configs = get_configs()
    samples = collect_raw_samples(args.raw_dir)
    split = stratified_group_split(samples, train_ratio=0.70, val_ratio=0.15, seed=args.seed)
    save_split_manifest(split, Path(args.out_dir) / "split_manifest.csv")

    print("Class order:", CLASS_NAMES, flush=True)
    print("Total images:", len(samples), flush=True)
    source_counts = defaultdict(int)
    for s in samples:
        source_counts[s["source"]] += 1
    print("Source counts:", dict(source_counts), flush=True)
    for split_name, items in split.items():
        counts = {cls: sum(s["class_name"] == cls for s in items) for cls in CLASS_NAMES}
        print(f"{split_name}: {len(items)} {counts}", flush=True)

    if args.run_all:
        summary = []
        best_name, best_score, best_path = None, -1.0, None
        for name, cfg in configs.items():
            score, path = train_one_config(name, cfg, split, args)
            summary.append({"config": name, "best_val_acc": score, "checkpoint": str(path)})
            if score > best_score:
                best_name, best_score, best_path = name, score, path
        Path(args.out_dir).mkdir(parents=True, exist_ok=True)
        with (Path(args.out_dir) / "experiment_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"Best experiment: {best_name} score={best_score:.4f} checkpoint={best_path}", flush=True)
    else:
        train_one_config(args.config, configs[args.config], split, args)

    if args.final_train_all:
        train_final_on_all(samples, configs[args.config], args, example_samples=split["test"])


if __name__ == "__main__":
    main()
