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
    return " ".join(LOGIT_TO_PHONEME[item] for item in collapsed)


def build_baseline(model_args):
    return GRUDecoder(
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


def build_proposed(model_args):
    return BaselineGRUWithAdapter(
        neural_dim=model_args.model.n_input_features,
        n_units=model_args.model.n_units,
        n_days=len(model_args.dataset.sessions),
        n_classes=model_args.dataset.n_classes,
        rnn_dropout=model_args.model.rnn_dropout,
        input_dropout=model_args.model.input_network.input_layer_dropout,
        n_layers=model_args.model.n_layers,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
        logit_adapter=True,
    )


def load_baseline(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.eval()
    return model


def lm_decode(r, logits):
    r.flushall()
    input_stream = "remote_lm_input"
    partial_stream = "remote_lm_output_partial"
    final_stream = "remote_lm_output_final"

    reset_seen = get_current_redis_time_ms(r)
    partial_seen = get_current_redis_time_ms(r)
    final_seen = get_current_redis_time_ms(r)

    reset_seen = reset_remote_language_model(r, reset_seen)
    lm_logits = rearrange_speech_logits_pt(logits[None, :, :])[0]
    partial_seen, partial = send_logits_to_remote_lm(
        r,
        input_stream,
        partial_stream,
        partial_seen,
        lm_logits,
    )
    final_seen, lm_out = finalize_remote_lm(r, final_stream, final_seen)
    candidates = lm_out["candidate_sentences"]
    return partial, candidates


def main():
    parser = argparse.ArgumentParser(description="Compara baseline y modelo propuesto en ensayos test.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--baseline_checkpoint", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"))
    parser.add_argument("--proposed_checkpoint", default=str(THIS_DIR / "salidas" / "baseline_adapter_logit" / "best_checkpoint.pt"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--session_index", type=int, default=1)
    parser.add_argument("--start_trial", type=int, default=0)
    parser.add_argument("--num_trials", type=int, default=10)
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)
    session = sessions[args.session_index]

    baseline = load_baseline(build_baseline(model_args), args.baseline_checkpoint)
    proposed = build_proposed(model_args)
    proposed_checkpoint = torch.load(args.proposed_checkpoint, map_location="cpu", weights_only=False)
    proposed.load_state_dict(proposed_checkpoint["model_state_dict"])
    proposed.eval()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    r.ping()

    hdf5_path = Path(args.data_dir) / session / "data_test.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        keys = list(h5_file.keys())
        selected_keys = keys[args.start_trial : args.start_trial + args.num_trials]

        print(f"Sesion: {session}")
        print(f"Split: test")
        print(f"Ensayos comparados: {len(selected_keys)}")
        print()
        print("| Trial | Baseline mejor frase | Propuesto mejor frase | Cambia |")
        print("|---|---|---|---|")

        details = []
        for trial_key in selected_keys:
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
                baseline_logits = baseline(neural, day_idx)[0].detach().numpy()
                proposed_logits = proposed(neural, day_idx)[0].detach().numpy()

            baseline_partial, baseline_candidates = lm_decode(r, baseline_logits)
            proposed_partial, proposed_candidates = lm_decode(r, proposed_logits)

            baseline_best = baseline_candidates[0]
            proposed_best = proposed_candidates[0]
            changed = "si" if baseline_best != proposed_best else "no"
            print(f"| {trial_key} | {baseline_best} | {proposed_best} | {changed} |")

            details.append(
                {
                    "trial": trial_key,
                    "baseline_phonemes": decode_argmax(baseline_logits),
                    "proposed_phonemes": decode_argmax(proposed_logits),
                    "baseline_partial": baseline_partial,
                    "proposed_partial": proposed_partial,
                    "baseline_candidates": baseline_candidates[:5],
                    "proposed_candidates": proposed_candidates[:5],
                }
            )

    print("\nDetalle por ensayo")
    for item in details:
        print(f"\n{item['trial']}")
        print(f"Baseline fonemas:  {item['baseline_phonemes']}")
        print(f"Propuesto fonemas: {item['proposed_phonemes']}")
        print(f"Baseline parcial:  {item['baseline_partial']}")
        print(f"Propuesto parcial: {item['proposed_partial']}")
        print("Top baseline:")
        for idx, sentence in enumerate(item["baseline_candidates"], start=1):
            print(f"  {idx}. {sentence}")
        print("Top propuesto:")
        for idx, sentence in enumerate(item["proposed_candidates"], start=1):
            print(f"  {idx}. {sentence}")


if __name__ == "__main__":
    main()
