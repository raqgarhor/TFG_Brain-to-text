import argparse
from pathlib import Path
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from datos import adjusted_lengths, ids_to_phonemes, smooth_batch  # noqa: E402


def format_phonemes(ids):
    return " ".join(ids_to_phonemes(ids))


def build_signal_image(original, smoothed, output_path, session, split, trial, feature):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_signal = original[:, feature]
    smoothed_signal = smoothed[:, feature]

    plt.figure(figsize=(12, 5))
    plt.plot(
        np.arange(len(original_signal)),
        original_signal,
        color="#1f77b4",
        linewidth=1.0,
        alpha=0.65,
        label="Senal original",
    )
    plt.plot(
        np.arange(len(smoothed_signal)),
        smoothed_signal,
        color="#f2a900",
        linewidth=2.2,
        label="Suavizado gaussiano",
    )
    plt.title(f"Senal neuronal en {session} - {split} - {trial} - caracteristica {feature}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor de la caracteristica neuronal")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Prepara los campos de un ensayo HDF5 para copiarlos en el panel admin."
    )
    parser.add_argument(
        "--args_path",
        default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"),
    )
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data" / "hdf5_data_final"),
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--feature", type=int, default=0)
    parser.add_argument(
        "--static_dir",
        default=str(REPO_ROOT.parents[0] / "brain-to-text-web" / "src" / "main" / "resources" / "static"),
        help="Carpeta static de la aplicacion web.",
    )
    parser.add_argument(
        "--crear_imagen",
        action="store_true",
        help="Genera una imagen de la senal en la carpeta static/images/signals.",
    )
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    hdf5_path = Path(args.data_dir) / args.session / f"data_{args.split}.hdf5"

    if not hdf5_path.exists():
        raise FileNotFoundError(f"No existe el archivo HDF5: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as h5_file:
        if args.trial not in h5_file:
            available = ", ".join(list(h5_file.keys())[:10])
            raise KeyError(f"No existe {args.trial}. Primeros ensayos disponibles: {available}")

        trial = h5_file[args.trial]
        features_np = trial["input_features"][:].astype(np.float32)
        features = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0)
        n_time_steps = int(trial.attrs["n_time_steps"])

        sentence = ""
        real_phonemes = ""
        if args.split in {"train", "val"}:
            sentence = str(trial.attrs.get("sentence_label", ""))
            if "seq_class_ids" in trial and "seq_len" in trial.attrs:
                seq_len = int(trial.attrs["seq_len"])
                ids = [int(item) for item in trial["seq_class_ids"][:seq_len]]
                real_phonemes = format_phonemes(ids)

    smoothed = smooth_batch(
        features,
        std=model_args.dataset.data_transforms.smooth_kernel_std,
        kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
        device="cpu",
    )
    input_len = adjusted_lengths(
        torch.tensor([n_time_steps]),
        smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
        smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )[0].item()

    image_name = f"{args.session}_{args.split}_{args.trial}_feature_{args.feature}.png"
    web_image_path = f"/images/signals/{image_name}"
    output_image_path = Path(args.static_dir) / "images" / "signals" / image_name

    if args.crear_imagen:
        if not 0 <= args.feature < features_np.shape[1]:
            raise ValueError(f"La caracteristica debe estar entre 0 y {features_np.shape[1] - 1}")
        build_signal_image(
            features_np,
            smoothed.squeeze(0).detach().cpu().numpy(),
            output_image_path,
            args.session,
            args.split,
            args.trial,
            args.feature,
        )

    print()
    print("Campos para crear el ensayo en el panel admin")
    print("--------------------------------------------")
    print(f"Sesion: {args.session}")
    print(f"Particion: {args.split}")
    print(f"Trial: {args.trial}")
    print(f"Frase real: {sentence if sentence else 'Sin frase disponible'}")
    print(f"Fonemas reales: {real_phonemes if real_phonemes else 'Sin fonemas reales disponibles'}")
    print(f"Forma input: {tuple(smoothed.shape)}")
    print(f"Forma logits: ({input_len}, {model_args.dataset.n_classes})")
    print(f"Ruta imagen: {web_image_path}")
    print("Notas: Ensayo real preparado desde HDF5 para cargarlo desde el panel de administracion.")
    print()
    if args.crear_imagen:
        print(f"Imagen creada en: {output_image_path}")
    else:
        print("Imagen no creada. Si la quieres generar, anade --crear_imagen.")
    print()


if __name__ == "__main__":
    main()
