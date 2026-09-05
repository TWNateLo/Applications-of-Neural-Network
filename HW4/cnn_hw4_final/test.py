from __future__ import annotations

# -----------------------------------------------------------------------------
# HW4 CNN Testing
# -----------------------------------------------------------------------------
#   1. How to run the test: (please place hidden test image route at --data_dir)
#   python test.py --data_dir hidden_test --weights models/best_model.pth --tta
#   2. If the tta processing is not allowed for hidden test, please run:
#   python test.py --data_dir hidden_test --weights models/best_model.pth
# -----------------------------------------------------------------------------

import argparse
import csv
import os
from pathlib import Path
from typing import List

# Fix common Windows Intel OpenMP duplicated DLL issue before torch/cv2 import.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
from PIL import Image, ImageOps
import torch
from torch import nn

torch.set_num_threads(1)

CLASS_NAMES = ["hao", "jin", "gua"]
SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


# =============================================================================
# 1. Preprocessing
# =============================================================================

def is_image_file(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in SUPPORTED_EXTS


def safe_open_image(path: str | Path) -> Image.Image:
    img = Image.open(path)
    return ImageOps.exif_transpose(img).convert("RGB")


def letterbox_resize(img: Image.Image, size: int) -> Image.Image:
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
    w, h = img.size
    crop_w, crop_h = int(w * margin_ratio), int(h * margin_ratio)
    left = max(0, (w - crop_w) // 2)
    top = max(0, (h - crop_h) // 2)
    return img.crop((left, top, left + crop_w, top + crop_h))


def upper_body_crop_view(img: Image.Image) -> Image.Image:
    w, h = img.size
    if h <= w:
        return center_crop_view(img, 0.92)
    return img.crop((0, 0, w, int(h * 0.72)))


def detect_face_crop(img: Image.Image, margin: float = 1.75) -> Image.Image | None:
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
    img = letterbox_resize(img, size)
    arr = np.asarray(img).astype(np.float32) / 255.0
    arr = (arr - 0.5) / 0.5
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def tta_tensors(path: str | Path, size: int, use_face_crop: bool = False) -> List[torch.Tensor]:
    img = safe_open_image(path)
    views: List[Image.Image] = [img, center_crop_view(img, 0.92), upper_body_crop_view(img)]
    if use_face_crop:
        face = detect_face_crop(img)
        if face is not None:
            views.append(face)
    return [image_to_tensor(view, size) for view in views]


# =============================================================================
# 2. CNN models - must match train.py
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


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        return ckpt
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt["model_state"] = ckpt["model"]
        return ckpt
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt["model_state"] = ckpt["state_dict"]
        return ckpt
    return {"model_state": ckpt, "arch": "compact", "class_names": CLASS_NAMES}


# =============================================================================
# 3. Testing / prediction
# =============================================================================

def collect_test_files(data_dir: str | Path):
    data_dir = Path(data_dir)
    samples = []
    has_class_folders = all((data_dir / cls).exists() for cls in CLASS_NAMES)

    if has_class_folders:
        for cls in CLASS_NAMES:
            for path in sorted((data_dir / cls).rglob("*")):
                if is_image_file(path):
                    samples.append((path, CLASS_NAMES.index(cls), cls))
    else:
        for path in sorted(data_dir.rglob("*")):
            if is_image_file(path):
                samples.append((path, None, None))

    if not samples:
        raise FileNotFoundError(f"No image files found in {data_dir}")
    return samples


def predict_one(model, path: Path, input_size: int, device, use_tta: bool, use_face_crop: bool = False):
    model.eval()
    with torch.no_grad():
        if use_tta:
            tensors = tta_tensors(path, input_size, use_face_crop=use_face_crop)
            batch = torch.stack(tensors, dim=0).to(device)
            logits = model(batch).mean(dim=0, keepdim=True)
        else:
            img = safe_open_image(path)
            tensor = image_to_tensor(img, input_size).unsqueeze(0).to(device)
            logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0].detach().cpu()
        pred_idx = int(probs.argmax())
        return pred_idx, probs.tolist()


def save_prediction_examples(rows: List[dict], out_path: Path):
    """Save a prediction contact sheet containing ALL prediction rows.

    HW4 only requires at least six examples, but this testing script is useful
    for checking every hidden/test prediction visually. Therefore, unlike
    train.py, test.py does not sample two examples per class; it includes every
    entry written to the prediction CSV.
    """
    if not rows:
        print("No prediction rows available; skipping prediction example image.", flush=True)
        return

    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"Skipping prediction example image because matplotlib is unavailable: {e}", flush=True)
        return

    chosen = rows  # Include all entries, not just two examples per class.
    n = len(chosen)
    if n <= 3:
        cols = n
    elif n <= 8:
        cols = 4
    else:
        cols = 5
    rows_n = (n + cols - 1) // cols

    plt.figure(figsize=(cols * 3.0, rows_n * 3.35))
    for i, row in enumerate(chosen, start=1):
        img = ImageOps.exif_transpose(Image.open(row["path"])).convert("RGB")
        # Downscale before plotting so large test folders do not create excessive RAM usage.
        img.thumbnail((320, 320), Image.Resampling.BICUBIC)

        plt.subplot(rows_n, cols, i)
        plt.imshow(img)
        plt.axis("off")

        true_text = row.get("true_label", "") or "unknown"
        pred_text = row["pred_label"]
        if true_text == "unknown":
            color = "black"
        else:
            color = "green" if true_text == pred_text else "red"

        # Show the row number so the image can be matched back to the CSV.
        plt.title(f"#{i}  True: {true_text}\nPred: {pred_text}", color=color, fontsize=9)

    plt.suptitle(f"Prediction examples - all {n} entries", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved all-entry prediction example image to {out_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True, help="Hidden test folder or normal test folder.")
    parser.add_argument("--weights", type=str, default="models/best_model.pth")
    parser.add_argument("--out_csv", type=str, default="outputs/test_predictions.csv")
    parser.add_argument("--examples_png", type=str, default="outputs/test_prediction_examples.png")
    parser.add_argument("--tta", action="store_true", help="Use full image + center/upper crop views and average predictions.")
    parser.add_argument("--face_crop_tta", action="store_true", help="Also use OpenCV face-crop TTA. Slower; use only if needed.")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    print("Starting test.py", flush=True)
    print(f"data_dir: {Path(args.data_dir).resolve()}", flush=True)
    print(f"weights: {Path(args.weights).resolve()}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ckpt = load_checkpoint(args.weights, map_location=device)
    class_names = ckpt.get("class_names", CLASS_NAMES)
    if list(class_names) != CLASS_NAMES:
        raise ValueError(f"Checkpoint class_names={class_names}, but code expects {CLASS_NAMES}. Do not change class order.")

    arch = ckpt.get("arch", "compact")
    input_size = int(ckpt.get("input_size", 128))
    base = int(ckpt.get("base", 24))
    dropout = float(ckpt.get("dropout", 0.35))
    model = create_model(arch, num_classes=len(CLASS_NAMES), base=base, dropout=dropout).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    samples = collect_test_files(args.data_dir)
    print(f"Found {len(samples)} test images. Device={device}, TTA={args.tta}, face_crop_tta={args.face_crop_tta}", flush=True)

    rows = []
    correct = 0
    total_with_labels = 0
    cm = torch.zeros(len(CLASS_NAMES), len(CLASS_NAMES), dtype=torch.long)

    for i, (path, true_idx, true_label) in enumerate(samples, start=1):
        pred_idx, probs = predict_one(model, path, input_size, device, use_tta=args.tta, use_face_crop=args.face_crop_tta)
        pred_label = CLASS_NAMES[pred_idx]
        if true_idx is not None:
            total_with_labels += 1
            correct += int(pred_idx == true_idx)
            cm[true_idx, pred_idx] += 1
        rows.append(
            {
                "path": str(path),
                "true_label": true_label if true_label is not None else "",
                "pred_label": pred_label,
                "prob_hao": probs[0],
                "prob_jin": probs[1],
                "prob_gua": probs[2],
            }
        )
        if i % 20 == 0 or i == len(samples):
            print(f"Predicted {i}/{len(samples)} images ...", flush=True)

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["path", "true_label", "pred_label", "prob_hao", "prob_jin", "prob_gua"])
        writer.writeheader()
        writer.writerows(rows)

    if total_with_labels:
        acc = correct / total_with_labels
        print(f"Accuracy: {acc:.4f} ({correct}/{total_with_labels})", flush=True)
        print("Confusion matrix rows=true, columns=pred, class order=", CLASS_NAMES, flush=True)
        print(cm.numpy(), flush=True)
    else:
        print(f"Predicted {len(rows)} images. No true labels were found, so accuracy was not computed.", flush=True)
    print(f"Saved predictions to {out_csv}", flush=True)
    save_prediction_examples(rows, Path(args.examples_png))


if __name__ == "__main__":
    main()
