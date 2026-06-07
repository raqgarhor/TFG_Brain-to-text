#!/usr/bin/env python3
import argparse
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_TRAINING = REPO_ROOT / "model_training"
sys.path.insert(0, str(MODEL_TRAINING))

from data_augmentations import gauss_smooth  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(
        description="Representa una característica neuronal antes y después del suavizado gaussiano."
    )
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data" / "t15_copyTask_neuralData" / "hdf5_data_final"),
        help="Carpeta hdf5_data_final del dataset.",
    )
    parser.add_argument("--session", default="t15.2023.08.13", help="Sesión a analizar.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"], help="Partición del dataset.")
    parser.add_argument("--trial", default="trial_0000", help="Ensayo dentro del archivo HDF5.")
    parser.add_argument(
        "--feature",
        type=int,
        default=0,
        help="Índice de la característica neuronal que se quiere representar.",
    )
    parser.add_argument("--window", type=int, default=100, help="Tamaño de la ventana gaussiana.")
    parser.add_argument("--std", type=float, default=2.0, help="Desviación típica de la gaussiana.")
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "pruebas" / "suavizado_gaussiano_trial.png"),
        help="Ruta donde se guardará la figura.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    hdf5_path = Path(args.data_dir) / args.session / f"data_{args.split}.hdf5"

    if not hdf5_path.exists():
        raise FileNotFoundError(f"No existe el archivo: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        if args.trial not in h5_file:
            available = ", ".join(list(h5_file.keys())[:5])
            raise KeyError(f"No existe {args.trial}. Primeros ensayos disponibles: {available}")

        trial_group = h5_file[args.trial]
        features = trial_group["input_features"][:].astype(np.float32)
        sentence = trial_group.attrs.get("sentence_label", "Sin frase disponible")

    if not 0 <= args.feature < features.shape[1]:
        raise ValueError(f"La característica debe estar entre 0 y {features.shape[1] - 1}")

    x = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
    smoothed = gauss_smooth(
        x,
        device="cpu",
        smooth_kernel_std=args.std,
        smooth_kernel_size=args.window,
        padding="valid",
    )
    smoothed = smoothed.squeeze(0).detach().cpu().numpy()

    original_signal = features[:, args.feature]
    smoothed_signal = smoothed[:, args.feature]

    original_time = np.arange(len(original_signal))
    smoothed_time = np.arange(len(smoothed_signal))

    plt.figure(figsize=(12, 5))
    plt.plot(original_time, original_signal, color="#1f77b4", linewidth=1.0, alpha=0.65, label="Señal original")
    plt.plot(smoothed_time, smoothed_signal, color="#f2a900", linewidth=2.2, label="Suavizado gaussiano")
    plt.title(f"Suavizado gaussiano en {args.session} - {args.trial} - característica {args.feature}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor de la característica neuronal")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200)

    print(f"Archivo HDF5: {hdf5_path}")
    print(f"Frase del ensayo: {sentence}")
    print(f"Forma original: {features.shape}")
    print(f"Forma tras suavizado: {smoothed.shape}")
    print(f"Ventana gaussiana: {args.window}")
    print(f"Desviación típica: {args.std}")
    print(f"Figura guardada en: {output_path}")


if __name__ == "__main__":
    main()
