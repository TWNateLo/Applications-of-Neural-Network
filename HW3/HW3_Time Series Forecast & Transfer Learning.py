import json
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


# ============================================================
# 0. Config
# ============================================================
@dataclass
class Config:
    seed: int = 42

    # Target-domain files
    train_file: str = "Anonymized_Train_Data.csv"
    test_file: str = "Anonymized_Inference_Data.csv"

    # Source-domain file
    source_file: str = "FeatureAndMetadata_Milling.csv"

    # HW3 hinted target windowing
    lookback: int = 90
    horizon: int = 60
    chunk_gap_minutes: int = 5

    # Source sequence windowing
    source_lookback: int = 20
    source_horizon: int = 1

    # Model
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Training
    batch_size: int = 256
    source_batch_size: int = 64
    scratch_epochs: int = 25
    source_epochs: int = 35
    finetune_epochs: int = 25
    lr_scratch: float = 1e-3
    lr_source: float = 1e-3
    lr_finetune: float = 8e-4
    weight_decay: float = 1e-5
    patience: int = 6
    val_ratio: float = 0.2

    # Optional quick-debug setting.
    # For final run, keep None. For fast testing, set to e.g. 12000.
    target_train_window_limit: int | None = None

    # Optional cleaning for the suspicious 200-target & broken-sensor rows.
    # Keep False if you want to preserve the exact HW3 hint preprocessing style.
    # Set True if you want to remove likely corrupted 200-target rows.
    # This is proven to be not helping at all after experimenting...
    bad_sensor_cleaning: bool = False

    output_dir: str = "hw3_current_only_outputs"


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Try to enable cuda if possible (or else use cpu)
def get_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def clone_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


# ============================================================
# 1. Target-domain preprocessing
# ============================================================
def get_target_feature_columns(train_path: str, test_path: str) -> List[str]:
    # The inference file may contain columns not in the train file.
    # Therefore, use only common columns for fair train/test preprocessing.
    train_cols = pd.read_csv(train_path, nrows=5).columns.tolist()
    test_cols = pd.read_csv(test_path, nrows=5).columns.tolist()
    common_cols = [c for c in train_cols if c in test_cols]

    # Assigning columns
    target_col = "Target_Value"
    time_col = "Timestamp"
    sequence_col = "Sequence_ID"
    cat_col = "Material_Composition"

    feature_columns = [
        c for c in common_cols
        if c not in [target_col, time_col, sequence_col, cat_col]
    ]
    return feature_columns


def preprocess_and_split_target(
    file_path: str,
    feature_columns: List[str],
    cat_col: str = "Material_Composition",
    target_col: str = "Target_Value",
    time_col: str = "Timestamp",
    sequence_col: str = "Sequence_ID",
    lookback: int = 90,
    horizon: int = 60,
    chunk_gap_minutes: int = 5,
    ohe: OneHotEncoder | None = None,
    scaler_x: StandardScaler | None = None,
    scaler_y: StandardScaler | None = None,
    fit: bool = True,
    bad_sensor_cleaning: bool = False,
) -> Tuple[np.ndarray, np.ndarray, OneHotEncoder, StandardScaler, StandardScaler, Dict]:
    """
    This follows the HW3 hinted preprocessing:
    - Convert target to numeric; Bad/Error -> NaN
    - Drop invalid target rows
    - Sort by Sequence_ID and Timestamp
    - Split continuous chunks by time gaps
    - Build X as past 90 steps
    - Build y as next 60 target values
    - Final metric uses y[:, 59]
    """
    df = pd.read_csv(file_path)

    keep_cols = [time_col, sequence_col, cat_col, target_col] + feature_columns
    keep_cols = [c for c in keep_cols if c in df.columns]
    df = df[keep_cols].copy()

    # --- 1. Handle non-numeric target labels, e.g., Bad / Error ---
    rows_before_target_clean = len(df)
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df = df.dropna(subset=[target_col]).reset_index(drop=True)
    rows_after_target_clean = len(df)

    # --- Optional conservative cleaning for suspicious "bad sensors (value = 0)" entries/rows ---
    # This is disabled in the final run due to its ineffectiveness (even making the results worse due to potential overly-aggressive data cleaning.)
    bad_sensor_rows_removed = 0
    if bad_sensor_cleaning:
        numeric_df = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        numeric_df = numeric_df.fillna(0)

        zero_count = (numeric_df == 0).sum(axis=1)
        minus_one_count = (numeric_df == -1).sum(axis=1)

        # Remove only rows where target is 200 AND sensors look obviously broken.
        suspicious_sensor_mask = (zero_count >= 2) | (minus_one_count >= 1)
        bad_200_mask = (df[target_col] == 200) & suspicious_sensor_mask

        bad_sensor_rows_removed = int(bad_200_mask.sum())
        df = df.loc[~bad_200_mask].copy().reset_index(drop=True)

    # --- 2. Handle discontinuous timestamps ---
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.sort_values([sequence_col, time_col]).reset_index(drop=True)

    time_diff = df.groupby(sequence_col)[time_col].diff().dt.total_seconds() / 60
    df["chunk_id"] = (
        (df[sequence_col] != df[sequence_col].shift(1)) |
        (time_diff > chunk_gap_minutes)
    ).cumsum()

    # --- 3. Feature processing ---
    X_num = df[feature_columns].apply(pd.to_numeric, errors="coerce")
    X_num = X_num.fillna(X_num.median(numeric_only=True)).fillna(0)

    if fit:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        X_cat = ohe.fit_transform(df[[cat_col]])
    else:
        X_cat = ohe.transform(df[[cat_col]])

    if fit:
        scaler_x = StandardScaler()
        X_num_scaled = scaler_x.fit_transform(X_num)
    else:
        X_num_scaled = scaler_x.transform(X_num)

    y_raw = df[[target_col]].values
    if fit:
        scaler_y = StandardScaler()
        y_scaled = scaler_y.fit_transform(y_raw).reshape(-1)
    else:
        y_scaled = scaler_y.transform(y_raw).reshape(-1)

    X_all = np.hstack([X_num_scaled, X_cat]).astype(np.float32)
    df["_y_scaled"] = y_scaled.astype(np.float32)

    # --- 4. Sliding windows inside each continuous chunk ---
    all_X, all_y = [], []

    for _, group in df.groupby("chunk_id", sort=True):
        if len(group) < lookback + horizon:
            continue

        idx = group.index.to_numpy()
        chunk_X = X_all[idx]
        chunk_y = group["_y_scaled"].to_numpy(dtype=np.float32)

        for i in range(len(group) - lookback - horizon + 1):
            # Input: previous 90 steps
            all_X.append(chunk_X[i:i + lookback])
            # Target: next 60 target values
            all_y.append(chunk_y[i + lookback:i + lookback + horizon])

    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.float32)

    stats = {
        "file": file_path,
        "rows_before_target_clean": int(rows_before_target_clean),
        "rows_after_target_clean": int(rows_after_target_clean),
        "bad_sensor_rows_removed": int(bad_sensor_rows_removed),
        "rows_after_all_cleaning": int(len(df)),
        "chunk_count": int(df["chunk_id"].nunique()),
        "window_count": int(len(X)),
        "feature_dim_after_encoding": int(X.shape[-1]) if len(X) else 0,
        "lookback": int(lookback),
        "horizon": int(horizon),
        "chunk_gap_minutes": int(chunk_gap_minutes),
    }

    return X, y, ohe, scaler_x, scaler_y, stats


# ============================================================
# 2. Source-domain preprocessing: ONLY current columns
# ============================================================
def load_source_feature_table(file_path: str) -> pd.DataFrame:
    """
    FeatureAndMetadata_Milling.csv uses:
    - semicolon separator
    - first data row as true headers
    - decimal commas in some columns
    """
    df = pd.read_csv(file_path, sep=";", dtype=str)
    df.columns = df.iloc[0]
    df = df.iloc[1:].reset_index(drop=True)

    for c in df.columns:
        df[c] = df[c].str.replace(",", ".", regex=False)

    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass

    return df


def get_current_only_source_columns(df: pd.DataFrame) -> List[str]:
    """
    Final source-feature selection based on my prior source domain feature experiment:
    use ONLY current-related source features.

    This includes current min/max/mean/std/skew/kurtosis columns from the
    source feature table, but excludes metadata and leakage columns.
    """
    leakage_or_helper = {
        "FileName",
        "CycleToFailure",
        "CycleToFailureNormalized",
        "NumberOfCycle",
        "TollIndex",
        "SampleIndex",
    }

    current_cols = [
        c for c in df.columns
        if c not in leakage_or_helper
        and pd.api.types.is_numeric_dtype(df[c])
        and "current" in str(c).lower()
    ]

    # Error/exception handling
    if not current_cols:
        raise ValueError("No current-related source feature columns were found.")

    return current_cols

# Source domain dataset preprocessing
def preprocess_and_split_source_current_only(
    file_path: str,
    lookback: int = 20,
    horizon: int = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Source sequence construction using only current-related columns:
    - group by TollIndex
    - sort by NumberOfCycle
    - X = previous source_lookback cycles of current-related features
    - y = future CycleToFailureNormalized scalar
    """
    df = load_source_feature_table(file_path)
    df = df.sort_values(["TollIndex", "NumberOfCycle"]).reset_index(drop=True)

    feature_columns = get_current_only_source_columns(df)

    scaler_x = StandardScaler()
    X_all = scaler_x.fit_transform(df[feature_columns]).astype(np.float32)

    scaler_y = StandardScaler()
    y_all = scaler_y.fit_transform(df[["CycleToFailureNormalized"]]).reshape(-1).astype(np.float32)

    all_X, all_y = [], []

    for _, group in df.groupby("TollIndex", sort=True):
        idx = group.index.to_numpy()

        if len(group) < lookback + horizon:
            continue

        group_X = X_all[idx]
        group_y = y_all[idx]

        for i in range(len(group) - lookback - horizon + 1):
            all_X.append(group_X[i:i + lookback])
            all_y.append(group_y[i + lookback + horizon - 1])

    X = np.asarray(all_X, dtype=np.float32)
    y = np.asarray(all_y, dtype=np.float32)

    split_idx = int(len(X) * 0.8)
    X_train, y_train = X[:split_idx], y[:split_idx]
    X_val, y_val = X[split_idx:], y[split_idx:]

    tool_lengths = df.groupby("TollIndex").size()

    stats = {
        "source_variant": "current_only",
        "source_rows": int(len(df)),
        "source_tools": int(df["TollIndex"].nunique()),
        "source_windows_total": int(len(X)),
        "source_windows_train": int(len(X_train)),
        "source_windows_val": int(len(X_val)),
        "source_feature_dim": int(X.shape[-1]) if len(X) else 0,
        "selected_feature_count": int(len(feature_columns)),
        "selected_features": feature_columns,
        "max_rows_per_tool": int(tool_lengths.max()),
        "median_rows_per_tool": float(tool_lengths.median()),
    }

    return X_train, y_train, X_val, y_val, stats


# ============================================================
# 3. Models
# ============================================================
class SourceLSTMRegressor(nn.Module):
    """
    Source model:
        input  shape: (batch, source_lookback, current_feature_dim)
        output shape: (batch,)
    """
    def __init__(self, input_dim: int, hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head_hidden = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.out = nn.Linear(hidden_size // 2, 1)

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        out, _ = self.lstm(x)
        final_state = out[:, -1, :]
        z = self.head_hidden(final_state)
        return self.out(z).squeeze(-1)


class TargetLSTMForecaster(nn.Module):
    """
    Target model:
        input  shape: (batch, 90, target_feature_dim)
        output shape: (batch, 60)
    """
    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 60,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, horizon),
        )

    def forward(self, x):
        x = self.relu(self.input_proj(x))
        out, _ = self.lstm(x)
        final_state = out[:, -1, :]
        return self.head(final_state)


def transfer_source_to_target(target_model: TargetLSTMForecaster, source_model: SourceLSTMRegressor):
    """
    Transfer source learned weights into target model.

    This transfers:
    - LSTM temporal encoder
    - first hidden regression head layer

    This does NOT transfer source input_proj because source and target feature
    dimensions are different.
    """
    source_state = source_model.state_dict()
    target_state = target_model.state_dict()

    for k, v in source_state.items():
        # Transfer LSTM weights
        if k.startswith("lstm.") and k in target_state and target_state[k].shape == v.shape:
            target_state[k] = v.clone()

        # Transfer source head_hidden.0 -> target head.0 if shape compatible
        if k.startswith("head_hidden.0"):
            target_key = k.replace("head_hidden.0", "head.0")
            if target_key in target_state and target_state[target_key].shape == v.shape:
                target_state[target_key] = v.clone()

    target_model.load_state_dict(target_state)


def apply_freeze_strategy(model: TargetLSTMForecaster, strategy: str):
    """
    Current final recommended strategy is full_finetune.
    Extra freeze strategies are included for report comparison.
    """
    for p in model.parameters():
        p.requires_grad = True

    if strategy == "scratch":
        return
    if strategy == "full_finetune":
        return
    if strategy == "freeze_all_feature_layers":
        for name, p in model.named_parameters():
            if name.startswith("lstm.") or name.startswith("head.0"):
                p.requires_grad = False
        return
    if strategy == "freeze_first_lstm_layer":
        prefixes = [
            "lstm.weight_ih_l0",
            "lstm.weight_hh_l0",
            "lstm.bias_ih_l0",
            "lstm.bias_hh_l0",
        ]
        for name, p in model.named_parameters():
            if any(name.startswith(prefix) for prefix in prefixes):
                p.requires_grad = False
        return

    raise ValueError(f"Unknown strategy: {strategy}")


# ============================================================
# 4. Training / evaluation
# ============================================================
def build_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    dataset = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[nn.Module, Dict[str, List[float]], int]:
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    wait = 0

    history = {"train_loss": [], "val_loss": []}

    for epoch in range(epochs):
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss_sum += loss.item() * len(xb)
            train_count += len(xb)

        train_loss = train_loss_sum / max(train_count, 1)
        history["train_loss"].append(train_loss)

        model.eval()
        val_loss_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)

                val_loss_sum += loss.item() * len(xb)
                val_count += len(xb)

        val_loss = val_loss_sum / max(val_count, 1)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch + 1:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch + 1
            best_state = clone_state_dict(model)
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch + 1}.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history, best_epoch


def evaluate_60th_step(
    model: TargetLSTMForecaster,
    X_samples: np.ndarray,
    y_samples: np.ndarray,
    scaler_y: StandardScaler,
    device: torch.device,
    batch_size: int = 512,
) -> Dict:
    """
    Exact HW3 hinted evaluation:
        y_pred_60th = model_output[:, 59]
        y_true_60th = y_samples[:, 59]
    """
    model.eval()
    loader = build_loader(X_samples, y_samples, batch_size=batch_size, shuffle=False)

    pred_list = []
    true_list = []

    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device)
            model_output = model(xb).cpu().numpy()

            y_pred_60th = model_output[:, 59]
            y_true_60th = yb.numpy()[:, 59]

            pred_list.append(y_pred_60th)
            true_list.append(y_true_60th)

    y_pred_60th = np.concatenate(pred_list)
    y_true_60th = np.concatenate(true_list)

    y_pred_rescaled = scaler_y.inverse_transform(y_pred_60th.reshape(-1, 1)).reshape(-1)
    y_true_rescaled = scaler_y.inverse_transform(y_true_60th.reshape(-1, 1)).reshape(-1)

    mae = mean_absolute_error(y_true_rescaled, y_pred_rescaled)
    mse = mean_squared_error(y_true_rescaled, y_pred_rescaled)
    mape = np.mean(
        np.abs((y_true_rescaled - y_pred_rescaled) / np.clip(np.abs(y_true_rescaled), 1e-6, None))
    ) * 100
    r2 = r2_score(y_true_rescaled, y_pred_rescaled)

    return {
        "mae": float(mae),
        "mse": float(mse),
        "mape": float(mape),
        "r2": float(r2),
        "y_pred_60th": y_pred_rescaled,
        "y_true_60th": y_true_rescaled,
    }


def save_prediction_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path):
    plt.figure(figsize=(14, 5))
    plt.plot(y_pred, label="Predicted", color="blue", linewidth=1.5)
    plt.scatter(np.arange(len(y_true)), y_true, label="Actual", color="red", marker="x", s=12)
    plt.xlabel("Sample")
    plt.ylabel("Target Value")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_convergence_plot(histories: Dict[str, Dict[str, List[float]]], save_path: Path):
    plt.figure(figsize=(12, 5))
    for name, history in histories.items():
        plt.plot(history["val_loss"], label=name)
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE Loss")
    plt.title("Loss Convergence Curve Comparison: Scratch vs Current-Only Transfer Learning")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


# ============================================================
# 5. Main experiment
# ============================================================
def main():
    config = Config()
    set_seed(config.seed)
    device = get_device()
    out_dir = ensure_dir(config.output_dir)

    print("=" * 90)
    print("HW3 Current-Only Source-Domain LSTM Transfer Learning")
    print(f"Device: {device}")
    print(f"Output directory: {out_dir.resolve()}")
    print("=" * 90)

    # -----------------------------
    # Target preprocessing
    # -----------------------------
    target_feature_columns = get_target_feature_columns(config.train_file, config.test_file)

    X_train_all, y_train_all, ohe, scaler_x, scaler_y, target_train_stats = preprocess_and_split_target(
        file_path=config.train_file,
        feature_columns=target_feature_columns,
        lookback=config.lookback,
        horizon=config.horizon,
        chunk_gap_minutes=config.chunk_gap_minutes,
        fit=True,
        bad_sensor_cleaning=config.bad_sensor_cleaning,
    )

    X_test, y_test, _, _, _, target_test_stats = preprocess_and_split_target(
        file_path=config.test_file,
        feature_columns=target_feature_columns,
        lookback=config.lookback,
        horizon=config.horizon,
        chunk_gap_minutes=config.chunk_gap_minutes,
        ohe=ohe,
        scaler_x=scaler_x,
        scaler_y=scaler_y,
        fit=False,
        bad_sensor_cleaning=config.bad_sensor_cleaning,
    )

    if config.target_train_window_limit is not None:
        limit = min(config.target_train_window_limit, len(X_train_all))
        X_train_all = X_train_all[:limit]
        y_train_all = y_train_all[:limit]

    split_idx = int(len(X_train_all) * (1 - config.val_ratio))
    X_target_train, y_target_train = X_train_all[:split_idx], y_train_all[:split_idx]
    X_target_val, y_target_val = X_train_all[split_idx:], y_train_all[split_idx:]

    target_train_loader = build_loader(X_target_train, y_target_train, config.batch_size, shuffle=True)
    target_val_loader = build_loader(X_target_val, y_target_val, config.batch_size * 2, shuffle=False)

    print("\nTarget preprocessing completed.")
    print(json.dumps(target_train_stats, indent=2))
    print(json.dumps(target_test_stats, indent=2))
    print("X_target_train:", X_target_train.shape)
    print("y_target_train:", y_target_train.shape)
    print("X_target_val:", X_target_val.shape)
    print("X_test:", X_test.shape)
    print("y_test:", y_test.shape)

    # -----------------------------
    # Source preprocessing: current only
    # -----------------------------
    Xs_train, ys_train, Xs_val, ys_val, source_stats = preprocess_and_split_source_current_only(
        file_path=config.source_file,
        lookback=config.source_lookback,
        horizon=config.source_horizon,
    )

    source_train_loader = build_loader(Xs_train, ys_train, config.source_batch_size, shuffle=True)
    source_val_loader = build_loader(Xs_val, ys_val, config.source_batch_size * 2, shuffle=False)

    print("\nSource preprocessing completed: CURRENT ONLY.")
    print(json.dumps({k: v for k, v in source_stats.items() if k != "selected_features"}, indent=2))
    print("Selected current source features:")
    for c in source_stats["selected_features"]:
        print("  -", c)

    # -----------------------------
    # 1. Scratch baseline
    # -----------------------------
    results = []
    histories = {}

    print("\n" + "=" * 90)
    print("Training scratch baseline")
    print("=" * 90)

    scratch_model = TargetLSTMForecaster(
        input_dim=X_target_train.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        horizon=config.horizon,
    )

    apply_freeze_strategy(scratch_model, "scratch")

    scratch_model, scratch_history, scratch_best_epoch = train_model(
        model=scratch_model,
        train_loader=target_train_loader,
        val_loader=target_val_loader,
        device=device,
        epochs=config.scratch_epochs,
        lr=config.lr_scratch,
        weight_decay=config.weight_decay,
        patience=config.patience,
    )

    scratch_metrics = evaluate_60th_step(scratch_model, X_test, y_test, scaler_y, device, batch_size=config.batch_size * 2)
    histories["scratch"] = scratch_history

    save_prediction_plot(
        y_true=scratch_metrics["y_true_60th"],
        y_pred=scratch_metrics["y_pred_60th"],
        title="Prediction Line Plot - Step 60 (scratch)",
        save_path=out_dir / "prediction_line_plot_scratch.png",
    )

    results.append({
        "experiment": "scratch",
        "source_variant": "none",
        "target_strategy": "scratch",
        "mae": scratch_metrics["mae"],
        "mse": scratch_metrics["mse"],
        "mape": scratch_metrics["mape"],
        "r2": scratch_metrics["r2"],
        "best_epoch": scratch_best_epoch,
        "source_feature_dim": None,
        "selected_feature_count": None,
        "source_windows_total": None,
        "trainable_params": int(sum(p.numel() for p in scratch_model.parameters() if p.requires_grad)),
        "total_params": int(sum(p.numel() for p in scratch_model.parameters())),
    })

    print("\nScratch 60th-step metrics:")
    print(json.dumps(results[-1], indent=2))

    # -----------------------------
    # 2. Source pretraining: current-only
    # -----------------------------
    print("\n" + "=" * 90)
    print("Pretraining source model with CURRENT-ONLY features")
    print("=" * 90)

    source_model = SourceLSTMRegressor(
        input_dim=Xs_train.shape[-1],
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )

    source_model, source_history, source_best_epoch = train_model(
        model=source_model,
        train_loader=source_train_loader,
        val_loader=source_val_loader,
        device=device,
        epochs=config.source_epochs,
        lr=config.lr_source,
        weight_decay=config.weight_decay,
        patience=config.patience,
    )

    histories["source_current_only_pretrain"] = source_history

    # -----------------------------
    # 3. Transfer learning: full fine-tune
    # -----------------------------
    transfer_strategies = [
        "full_finetune",
        "freeze_all_feature_layers",
        "freeze_first_lstm_layer",
    ]

    for strategy in transfer_strategies:
        print("\n" + "=" * 90)
        print(f"Transfer learning with current-only source: {strategy}")
        print("=" * 90)

        target_model = TargetLSTMForecaster(
            input_dim=X_target_train.shape[-1],
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout,
            horizon=config.horizon,
        )

        transfer_source_to_target(target_model, source_model)
        apply_freeze_strategy(target_model, strategy)

        target_model, history, best_epoch = train_model(
            model=target_model,
            train_loader=target_train_loader,
            val_loader=target_val_loader,
            device=device,
            epochs=config.finetune_epochs,
            lr=config.lr_finetune,
            weight_decay=config.weight_decay,
            patience=config.patience,
        )

        metrics = evaluate_60th_step(target_model, X_test, y_test, scaler_y, device, batch_size=config.batch_size * 2)
        exp_name = f"tl_current_only_{strategy}"
        histories[exp_name] = history

        save_prediction_plot(
            y_true=metrics["y_true_60th"],
            y_pred=metrics["y_pred_60th"],
            title=f"Prediction Line Plot - Step 60 ({exp_name})",
            save_path=out_dir / f"prediction_line_plot_{exp_name}.png",
        )

        result_row = {
            "experiment": exp_name,
            "source_variant": "current_only",
            "target_strategy": strategy,
            "mae": metrics["mae"],
            "mse": metrics["mse"],
            "mape": metrics["mape"],
            "r2": metrics["r2"],
            "best_epoch": best_epoch,
            "source_best_epoch": source_best_epoch,
            "source_feature_dim": source_stats["source_feature_dim"],
            "selected_feature_count": source_stats["selected_feature_count"],
            "source_windows_total": source_stats["source_windows_total"],
            "trainable_params": int(sum(p.numel() for p in target_model.parameters() if p.requires_grad)),
            "total_params": int(sum(p.numel() for p in target_model.parameters())),
        }

        results.append(result_row)

        print("\n60th-step metrics:")
        print(json.dumps(result_row, indent=2))

    # -----------------------------
    # Save outputs
    # -----------------------------
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(["mape", "mae"], ascending=[True, True])

    results_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    with open(out_dir / "metrics_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open(out_dir / "preprocessing_stats.json", "w", encoding="utf-8") as f:
        json.dump({
            "config": asdict(config),
            "target_train_stats": target_train_stats,
            "target_test_stats": target_test_stats,
            "source_stats_current_only": source_stats,
        }, f, indent=2)

    save_convergence_plot(histories, out_dir / "convergence_comparison.png")

    print("\n" + "=" * 90)
    print("Final metrics summary")
    print("=" * 90)
    print(results_df.to_string(index=False))

    print("\nSaved outputs:")
    print(f"- {out_dir / 'metrics_summary.csv'}")
    print(f"- {out_dir / 'metrics_summary.json'}")
    print(f"- {out_dir / 'preprocessing_stats.json'}")
    print(f"- {out_dir / 'convergence_comparison.png'}")
    print(f"- {out_dir / 'prediction_line_plot_*.png'}")


if __name__ == "__main__":
    main()
