import argparse
import os
import sys

import h5py
import numpy as np
import torch
from omegaconf import OmegaConf


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINING_DIR = os.path.join(PROJECT_DIR, "model_training")
sys.path.insert(0, MODEL_TRAINING_DIR)

from data_augmentations import gauss_smooth  # noqa: E402
from evaluate_model_helpers import LOGIT_TO_PHONEME  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402


def decode_argmax(logits):
    seq = np.argmax(logits, axis=-1)
    collapsed = []
    previous = None
    for item in seq:
        item = int(item)
        if item != 0 and item != previous:
            collapsed.append(item)
        previous = item
    return [LOGIT_TO_PHONEME[item] for item in collapsed]


def main():
    parser = argparse.ArgumentParser(
        description="Carga el baseline RNN preentrenado y ejecuta una inferencia corta."
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(
            PROJECT_DIR,
            "data",
            "t15_pretrained_rnn_baseline",
            "t15_pretrained_rnn_baseline",
        ),
        help="Carpeta del modelo preentrenado.",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(
            PROJECT_DIR,
            "data",
            "t15_copyTask_neuralData",
            "hdf5_data_final",
        ),
        help="Carpeta hdf5_data_final.",
    )
    parser.add_argument("--session_index", type=int, default=1)
    parser.add_argument("--trial_index", type=int, default=0)
    args = parser.parse_args()

    model_args = OmegaConf.load(os.path.join(args.model_path, "checkpoint", "args.yaml"))

    model = GRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_units=model_args.model.n_units,
        n_days=len(model_args.dataset.sessions),
        n_classes=model_args.dataset.n_classes,
        rnn_dropout=model_args.model.rnn_dropout,
        input_dropout=model_args.model.input_network.input_layer_dropout,
        n_layers=model_args.model.n_layers,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )

    checkpoint_path = os.path.join(args.model_path, "checkpoint", "best_checkpoint")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()

    session = model_args.dataset.sessions[args.session_index]
    hdf5_path = os.path.join(args.data_dir, session, "data_val.hdf5")

    with h5py.File(hdf5_path, "r") as h5_file:
        trial_key = list(h5_file.keys())[args.trial_index]
        trial = h5_file[trial_key]

        neural = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        day_idx = torch.tensor([args.session_index])

        neural = gauss_smooth(
            inputs=neural,
            device="cpu",
            smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
            smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            padding="valid",
        )

        with torch.no_grad():
            logits = model(neural, day_idx)[0].detach().numpy()

        predicted = decode_argmax(logits)
        true_ids = trial["seq_class_ids"][: trial.attrs["seq_len"]]
        true = [LOGIT_TO_PHONEME[int(item)] for item in true_ids]

        print(f"Sesion: {session}")
        print(f"Trial: {trial_key}")
        print(f"Forma input_features tras suavizado: {tuple(neural.shape)}")
        print(f"Forma logits: {tuple(logits.shape)}")
        print()
        print(f"Frase real: {trial.attrs['sentence_label']}")
        print(f"Fonemas reales: {' '.join(true)}")
        print(f"Fonemas predichos: {' '.join(predicted)}")
        print()
        print(f"Checkpoint val_PER: {checkpoint['val_PER']:.4f}")


if __name__ == "__main__":
    main()
