import os
import sys
from pathlib import Path

import h5py
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from data_augmentations import gauss_smooth  # noqa: E402


LOGIT_TO_PHONEME = [
    "BLANK",
    "AA", "AE", "AH", "AO", "AW",
    "AY", "B", "CH", "D", "DH",
    "EH", "ER", "EY", "F", "G",
    "HH", "IH", "IY", "JH", "K",
    "L", "M", "N", "NG", "OW",
    "OY", "P", "R", "S", "SH",
    "T", "TH", "UH", "UW", "V",
    "W", "Y", "Z", "ZH",
    " | ",
]


class HDF5TrialDataset(Dataset):
    def __init__(self, data_dir, sessions, split, session_to_day, max_trials_per_session=None):
        self.items = []

        for session in sessions:
            hdf5_path = Path(data_dir) / session / f"data_{split}.hdf5"
            if not hdf5_path.exists():
                continue

            with h5py.File(hdf5_path, "r") as h5_file:
                keys = list(h5_file.keys())

            if max_trials_per_session is not None:
                keys = keys[:max_trials_per_session]

            for key in keys:
                self.items.append(
                    {
                        "hdf5_path": str(hdf5_path),
                        "trial_key": key,
                        "day_idx": session_to_day[session],
                    }
                )

        if len(self.items) == 0:
            raise ValueError(f"No se encontraron ensayos para split={split} en {data_dir}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        with h5py.File(item["hdf5_path"], "r") as h5_file:
            trial = h5_file[item["trial_key"]]
            features = torch.tensor(trial["input_features"][:], dtype=torch.float32)
            labels = torch.tensor(trial["seq_class_ids"][:], dtype=torch.long)
            seq_len = int(trial.attrs["seq_len"])
            n_time_steps = int(trial.attrs["n_time_steps"])

        return {
            "features": features,
            "labels": labels,
            "seq_len": seq_len,
            "n_time_steps": n_time_steps,
            "day_idx": item["day_idx"],
            "trial_key": item["trial_key"],
            "hdf5_path": item["hdf5_path"],
        }


def collate_trials(batch):
    return {
        "features": pad_sequence([item["features"] for item in batch], batch_first=True, padding_value=0.0),
        "labels": pad_sequence([item["labels"] for item in batch], batch_first=True, padding_value=0),
        "seq_lens": torch.tensor([item["seq_len"] for item in batch], dtype=torch.long),
        "n_time_steps": torch.tensor([item["n_time_steps"] for item in batch], dtype=torch.long),
        "day_idx": torch.tensor([item["day_idx"] for item in batch], dtype=torch.long),
        "trial_key": [item["trial_key"] for item in batch],
        "hdf5_path": [item["hdf5_path"] for item in batch],
    }


def smooth_batch(features, std=2, kernel_size=100, device="cpu"):
    return gauss_smooth(
        inputs=features,
        device=device,
        smooth_kernel_std=std,
        smooth_kernel_size=kernel_size,
        padding="valid",
    )


def adjusted_lengths(n_time_steps, smooth_kernel_size=100, smooth_kernel_std=2, patch_size=14, patch_stride=4):
    # El kernel efectivo del repositorio conserva solo pesos gaussianos > 0.01.
    # Con std=2, ese kernel efectivo tiene longitud 9 y el suavizado valid reduce T en 8.
    # Para otros parametros, esta aproximacion usa el comportamiento habitual observado en el baseline.
    if smooth_kernel_size == 100 and smooth_kernel_std == 2:
        smoothed_lengths = n_time_steps - 8
    else:
        smoothed_lengths = n_time_steps
    return ((smoothed_lengths - patch_size) // patch_stride + 1).to(torch.long)


def decode_argmax(logits):
    seq = torch.argmax(logits, dim=-1).detach().cpu().tolist()
    collapsed = []
    previous = None
    for item in seq:
        if item != 0 and item != previous:
            collapsed.append(item)
        previous = item
    return collapsed


def ids_to_phonemes(ids):
    return [LOGIT_TO_PHONEME[int(item)] for item in ids]


def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(
                dp[j] + 1,
                dp[j - 1] + 1,
                prev + (0 if ca == cb else 1),
            )
            prev = old
    return dp[-1]


def parse_sessions(all_sessions, session_arg, max_sessions):
    if session_arg:
        sessions = [s.strip() for s in session_arg.split(",") if s.strip()]
    else:
        sessions = list(all_sessions)

    if max_sessions is not None:
        sessions = sessions[:max_sessions]

    missing = [s for s in sessions if s not in all_sessions]
    if missing:
        raise ValueError(f"Sesiones no encontradas en args.yaml: {missing}")

    return sessions


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
