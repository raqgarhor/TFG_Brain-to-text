import argparse
import os
import sys

import h5py
import numpy as np
import redis
import torch
from omegaconf import OmegaConf


PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_TRAINING_DIR = os.path.join(PROJECT_DIR, "model_training")
sys.path.insert(0, MODEL_TRAINING_DIR)

from data_augmentations import gauss_smooth  # noqa: E402
from evaluate_model_helpers import (  # noqa: E402
    LOGIT_TO_PHONEME,
    finalize_remote_lm,
    get_current_redis_time_ms,
    rearrange_speech_logits_pt,
    reset_remote_language_model,
    send_logits_to_remote_lm,
)
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
        description=(
            "Ejecuta un unico ensayo con la RNN preentrenada y pide texto final "
            "al language model remoto mediante Redis."
        )
    )
    parser.add_argument(
        "--model_path",
        default=os.path.join(PROJECT_DIR, "data", "t15_pretrained_rnn_baseline"),
        help="Carpeta del modelo RNN preentrenado.",
    )
    parser.add_argument(
        "--data_dir",
        default=os.path.join(PROJECT_DIR, "data", "hdf5_data_final"),
        help="Carpeta hdf5_data_final.",
    )
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--session_index", type=int, default=1)
    parser.add_argument("--trial_index", type=int, default=0)
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
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
    hdf5_path = os.path.join(args.data_dir, session, f"data_{args.split}.hdf5")

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

        pred_phonemes = decode_argmax(logits)

        print(f"Sesion: {session}")
        print(f"Split: {args.split}")
        print(f"Trial: {trial_key}")
        print(f"Forma input_features tras suavizado: {tuple(neural.shape)}")
        print(f"Forma logits RNN: {tuple(logits.shape)}")
        print(f"Fonemas RNN por argmax: {' '.join(pred_phonemes)}")

        if args.split == "val":
            true_ids = trial["seq_class_ids"][: trial.attrs["seq_len"]]
            true = [LOGIT_TO_PHONEME[int(item)] for item in true_ids]
            print(f"Frase real: {trial.attrs['sentence_label']}")
            print(f"Fonemas reales: {' '.join(true)}")

    print("\nConectando con Redis/modelo de lenguaje...")
    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    r.ping()
    r.flushall()

    remote_lm_input_stream = "remote_lm_input"
    remote_lm_output_partial_stream = "remote_lm_output_partial"
    remote_lm_output_final_stream = "remote_lm_output_final"

    reset_seen = get_current_redis_time_ms(r)
    partial_seen = get_current_redis_time_ms(r)
    final_seen = get_current_redis_time_ms(r)

    reset_seen = reset_remote_language_model(r, reset_seen)

    lm_logits = rearrange_speech_logits_pt(logits[None, :, :])[0]
    partial_seen, partial = send_logits_to_remote_lm(
        r,
        remote_lm_input_stream,
        remote_lm_output_partial_stream,
        partial_seen,
        lm_logits,
    )
    final_seen, lm_out = finalize_remote_lm(
        r,
        remote_lm_output_final_stream,
        final_seen,
    )

    print("\nTexto parcial del LM:")
    print(partial)
    print("\nMejor frase final:")
    print(lm_out["candidate_sentences"][0])
    print("\nTop candidatos:")
    for idx, sentence in enumerate(lm_out["candidate_sentences"][:5], start=1):
        print(f"{idx}. {sentence}")


if __name__ == "__main__":
    main()
