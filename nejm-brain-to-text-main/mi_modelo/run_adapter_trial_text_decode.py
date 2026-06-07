import argparse
from pathlib import Path
import sys

import h5py
import numpy as np
import redis
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from datos import smooth_batch  # noqa: E402
from evaluate_model_helpers import (  # noqa: E402
    LOGIT_TO_PHONEME,
    finalize_remote_lm,
    get_current_redis_time_ms,
    rearrange_speech_logits_pt,
    reset_remote_language_model,
    send_logits_to_remote_lm,
)
from modelo_baseline_adapter import BaselineGRUWithAdapter  # noqa: E402


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
        description="Ejecuta un ensayo con el modelo propuesto y obtiene texto con el LM por Redis."
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
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--session_index", type=int, default=1)
    parser.add_argument("--trial_index", type=int, default=0)
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)
    session = sessions[args.session_index]

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

    hdf5_path = Path(args.data_dir) / session / f"data_{args.split}.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        trial_key = list(h5_file.keys())[args.trial_index]
        trial = h5_file[trial_key]

        neural = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        day_idx = torch.tensor([args.session_index])

        neural = smooth_batch(
            neural,
            std=model_args.dataset.data_transforms.smooth_kernel_std,
            kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            device="cpu",
        )

        with torch.no_grad():
            logits = model(neural, day_idx)[0].detach().numpy()

        pred_phonemes = decode_argmax(logits)

        print(f"Sesion: {session}")
        print(f"Split: {args.split}")
        print(f"Trial: {trial_key}")
        print(f"Forma input_features tras suavizado: {tuple(neural.shape)}")
        print(f"Forma logits modelo propuesto: {tuple(logits.shape)}")
        print(f"Fonemas modelo propuesto por argmax: {' '.join(pred_phonemes)}")

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

    print()
    print(f"PER del checkpoint propuesto: {checkpoint.get('val_PER', '-')}")


if __name__ == "__main__":
    main()
