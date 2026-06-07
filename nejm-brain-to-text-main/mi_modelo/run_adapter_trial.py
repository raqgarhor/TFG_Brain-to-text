import argparse
from pathlib import Path
import sys

import h5py
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from datos import adjusted_lengths, ids_to_phonemes, smooth_batch  # noqa: E402
from modelo_baseline_adapter import BaselineGRUWithAdapter  # noqa: E402


def format_phonemes(ids):
    return " ".join(ids_to_phonemes(ids))


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta el modelo propuesto sobre un ensayo concreto."
    )
    parser.add_argument(
        "--args_path",
        default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"),
    )
    parser.add_argument(
        "--checkpoint",
        default=str(THIS_DIR / "salidas" / "baseline_adapter_logit" / "best_checkpoint.pt"),
    )
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data" / "hdf5_data_final"),
    )
    parser.add_argument("--session", default="t15.2023.08.13")
    parser.add_argument("--split", choices=["train", "val", "test"], default="val")
    parser.add_argument("--trial", default="trial_0000")
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)
    day_idx = sessions.index(args.session)

    model = BaselineGRUWithAdapter(
        neural_dim=model_args.model.n_input_features,
        n_units=model_args.model.n_units,
        n_days=len(sessions),
        n_classes=model_args.dataset.n_classes,
        rnn_dropout=model_args.model.rnn_dropout,
        input_dropout=model_args.model.input_network.input_layer_dropout,
        n_layers=model_args.model.n_layers,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
        logit_adapter=True,
    )

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    hdf5_path = Path(args.data_dir) / args.session / f"data_{args.split}.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        trial = h5_file[args.trial]
        features = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        n_time_steps = int(trial.attrs["n_time_steps"])
        sentence = trial.attrs.get("sentence_label", None)

        labels = None
        seq_len = None
        if "seq_class_ids" in trial:
            labels = torch.tensor(trial["seq_class_ids"][:], dtype=torch.long)
            seq_len = int(trial.attrs["seq_len"])

    smoothed = smooth_batch(
        features,
        std=model_args.dataset.data_transforms.smooth_kernel_std,
        kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
        device="cpu",
    )

    with torch.no_grad():
        logits = model(smoothed, torch.tensor([day_idx]))

    input_len = adjusted_lengths(
        torch.tensor([n_time_steps]),
        smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
        smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )[0].item()

    pred = torch.argmax(logits[0, :input_len, :], dim=-1)
    pred = torch.unique_consecutive(pred).cpu().tolist()
    pred = [item for item in pred if item != 0]

    print(f"Sesion: {args.session}")
    print(f"Split: {args.split}")
    print(f"Trial: {args.trial}")
    if sentence is not None:
        print(f"Frase real: {sentence}")
    print(f"Forma input_features tras suavizado: {tuple(smoothed.shape)}")
    print(f"Forma logits modelo propuesto: {tuple(logits[0].shape)}")
    print()
    print(f"Fonemas predichos modelo propuesto: {format_phonemes(pred)}")

    if labels is not None and seq_len is not None:
        true = labels[:seq_len].cpu().tolist()
        print(f"Fonemas reales: {format_phonemes(true)}")

    print()
    print(f"PER del checkpoint propuesto: {checkpoint.get('val_PER', '-')}")


if __name__ == "__main__":
    main()
