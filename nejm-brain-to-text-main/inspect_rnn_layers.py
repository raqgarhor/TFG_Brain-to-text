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


def summarize_tensor(name, tensor):
    arr = tensor.detach().cpu().numpy()
    flat = arr.reshape(-1)
    sample = ", ".join(f"{value:.4f}" for value in flat[:8])
    print(f"{name}")
    print(f"  forma: {tuple(arr.shape)}")
    print(f"  media: {float(flat.mean()):.4f}")
    print(f"  desviacion tipica: {float(flat.std()):.4f}")
    print(f"  minimo: {float(flat.min()):.4f}")
    print(f"  maximo: {float(flat.max()):.4f}")
    print(f"  primeros valores: [{sample}]")
    print()


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
        description="Inspecciona las salidas intermedias del modelo RNN baseline."
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_DIR, "data", "t15_pretrained_rnn_baseline"),
        help="Carpeta del modelo preentrenado.",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(PROJECT_DIR, "data", "hdf5_data_final"),
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
        neural_raw = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        sentence = trial.attrs["sentence_label"]
        true_ids = trial["seq_class_ids"][: trial.attrs["seq_len"]]
        true_phonemes = [LOGIT_TO_PHONEME[int(item)] for item in true_ids]

    day_idx = torch.tensor([args.session_index])

    print(f"Sesion: {session}")
    print(f"Trial: {trial_key}")
    print(f"Frase real: {sentence}")
    print()
    print("CONFIGURACION DEL MODELO")
    print(f"  n_input_features: {model_args.model.n_input_features}")
    print(f"  n_layers GRU: {model_args.model.n_layers}")
    print(f"  hidden_size GRU: {model_args.model.n_units}")
    print(f"  n_classes: {model_args.dataset.n_classes}")
    print(f"  n_sessions: {len(model_args.dataset.sessions)}")
    print(f"  input_dropout: {model_args.model.input_network.input_layer_dropout}")
    print(f"  rnn_dropout: {model_args.model.rnn_dropout}")
    print(f"  patch_size: {model_args.model.patch_size}")
    print(f"  patch_stride: {model_args.model.patch_stride}")
    print()

    with torch.no_grad():
        summarize_tensor("0. input_features original", neural_raw)

        x = gauss_smooth(
            inputs=neural_raw,
            device="cpu",
            smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
            smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            padding="valid",
        )
        summarize_tensor("1. entrada tras suavizado gaussiano", x)

        day_weights = torch.stack([model.day_weights[i] for i in day_idx], dim=0)
        day_biases = torch.cat([model.day_biases[i] for i in day_idx], dim=0).unsqueeze(1)
        x_adapted = torch.einsum("btd,bdk->btk", x, day_weights) + day_biases
        summarize_tensor("2. salida de adaptacion especifica por sesion", x_adapted)

        x_activated = model.day_layer_activation(x_adapted)
        summarize_tensor("3. salida tras Softsign", x_activated)

        x_dropout = model.day_layer_dropout(x_activated)
        summarize_tensor("4. salida tras dropout de entrada (en eval no cambia)", x_dropout)

        x_patched = x_dropout.unsqueeze(1)
        x_patched = x_patched.permute(0, 3, 1, 2)
        x_unfold = x_patched.unfold(3, model.patch_size, model.patch_stride)
        x_unfold = x_unfold.squeeze(2)
        x_unfold = x_unfold.permute(0, 2, 3, 1)
        x_patched = x_unfold.reshape(x_dropout.size(0), x_unfold.size(1), -1)
        summarize_tensor("5. salida tras agrupacion temporal", x_patched)

        initial_state = model.h0.expand(model.n_layers, x_patched.shape[0], model.n_units).contiguous()
        summarize_tensor("6. estado oculto inicial de la GRU", initial_state)

        gru_output, hidden_states = model.gru(x_patched, initial_state)
        summarize_tensor("7. salida temporal de la GRU", gru_output)
        summarize_tensor("8. estados ocultos finales de las 5 capas GRU", hidden_states)

        logits = model.out(gru_output)
        summarize_tensor("9. logits foneticos finales", logits)

    logits_np = logits[0].detach().cpu().numpy()
    predicted = decode_argmax(logits_np)

    print("DECODIFICACION CTC SIMPLE POR ARGMAX")
    print(f"  forma logits de un ensayo: {tuple(logits_np.shape)}")
    print(f"  pasos temporales de salida: {logits_np.shape[0]}")
    print(f"  clases por paso temporal: {logits_np.shape[1]}")
    print(f"  fonemas reales: {' '.join(true_phonemes)}")
    print(f"  fonemas predichos: {' '.join(predicted)}")
    print()
    print(f"Checkpoint val_PER: {checkpoint['val_PER']:.4f}")


if __name__ == "__main__":
    main()
