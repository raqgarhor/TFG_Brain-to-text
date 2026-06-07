import argparse
from pathlib import Path
import sys

import h5py
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))

from datos import adjusted_lengths, count_parameters, smooth_batch  # noqa: E402
from modelo_gru_compacto import CompactGRUDecoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Prueba rapida de formas del modelo GRU compacto.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--session", default="t15.2023.08.13")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--trial", default="trial_0000")
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)
    day_idx = sessions.index(args.session)

    model = CompactGRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_days=len(sessions),
        n_classes=model_args.dataset.n_classes,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )
    model.eval()

    hdf5_path = Path(args.data_dir) / args.session / f"data_{args.split}.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        trial = h5_file[args.trial]
        features = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        n_time_steps = torch.tensor([trial.attrs["n_time_steps"]])

    with torch.no_grad():
        smoothed = smooth_batch(
            features,
            std=model_args.dataset.data_transforms.smooth_kernel_std,
            kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            device="cpu",
        )
        logits = model(smoothed, torch.tensor([day_idx]))

    total_params, trainable_params = count_parameters(model)
    lengths = adjusted_lengths(
        n_time_steps,
        smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
        smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )

    print("Modelo propuesto: GRU compacto con proyeccion previa")
    print(f"Sesion: {args.session}")
    print(f"Split: {args.split}")
    print(f"Trial: {args.trial}")
    print(f"Forma input_features original: {tuple(features.shape)}")
    print(f"Forma tras suavizado: {tuple(smoothed.shape)}")
    print(f"Longitud ajustada para CTC: {int(lengths[0])}")
    print(f"Forma logits: {tuple(logits.shape)}")
    print(f"Parametros totales: {total_params:,}")
    print(f"Parametros entrenables: {trainable_params:,}")


if __name__ == "__main__":
    main()
