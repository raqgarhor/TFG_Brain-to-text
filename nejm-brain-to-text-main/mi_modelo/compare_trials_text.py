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

from datos import adjusted_lengths, edit_distance, ids_to_phonemes, smooth_batch  # noqa: E402
from evaluate_model_helpers import (  # noqa: E402
    finalize_remote_lm,
    get_current_redis_time_ms,
    rearrange_speech_logits_pt,
    reset_remote_language_model,
    send_logits_to_remote_lm,
)
from modelo_baseline_adapter import BaselineGRUWithAdapter  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402


def normalize_words(text):
    if not text:
        return []
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [word for word in cleaned.split() if word]


def decode_argmax_ids(logits):
    seq = np.argmax(logits, axis=-1)
    collapsed = []
    previous = None
    for item in seq:
        item = int(item)
        if item != 0 and item != previous:
            collapsed.append(item)
        previous = item
    return collapsed


def phoneme_text(ids):
    return " ".join(ids_to_phonemes(ids))


def format_metric(value):
    if value is None:
        return "-"
    return f"{value:.4f}"


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
    return partial, [str(item) for item in lm_out["candidate_sentences"]]


def compute_metrics(pred_ids, true_ids, pred_sentence, true_sentence):
    per = None
    if true_ids:
        per = edit_distance(pred_ids, true_ids) / len(true_ids)

    wer = None
    ref_words = normalize_words(true_sentence)
    hyp_words = normalize_words(pred_sentence)
    if ref_words:
        wer = edit_distance(hyp_words, ref_words) / len(ref_words)

    return per, wer


def main():
    parser = argparse.ArgumentParser(description="Compara baseline y modelo propuesto en ensayos val o test.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--baseline_checkpoint", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"))
    parser.add_argument("--proposed_checkpoint", default=str(THIS_DIR / "salidas" / "baseline_adapter_logit" / "best_checkpoint.pt"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
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

    hdf5_path = Path(args.data_dir) / session / f"data_{args.split}.hdf5"
    details = []
    totals = {
        "baseline_per_edits": 0,
        "baseline_wer_edits": 0,
        "proposed_per_edits": 0,
        "proposed_wer_edits": 0,
        "phoneme_len": 0,
        "word_len": 0,
    }

    with h5py.File(hdf5_path, "r") as h5_file:
        keys = list(h5_file.keys())
        selected_keys = keys[args.start_trial : args.start_trial + args.num_trials]

        print(f"Sesion: {session}")
        print(f"Split: {args.split}")
        print(f"Ensayos comparados: {len(selected_keys)}")
        print()
        print("| Trial | Frase real | Baseline mejor frase | Propuesto mejor frase | PER baseline | PER propuesto | WER baseline | WER propuesto | Cambia |")
        print("|---|---|---|---|---:|---:|---:|---:|---|")

        for trial_key in selected_keys:
            trial = h5_file[trial_key]
            neural = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
            n_time_steps = int(trial.attrs["n_time_steps"])
            day_idx = torch.tensor([args.session_index])
            sentence = trial.attrs.get("sentence_label", None)

            true_ids = None
            if "seq_class_ids" in trial:
                seq_len = int(trial.attrs["seq_len"])
                true_ids = [int(item) for item in trial["seq_class_ids"][:seq_len]]

            neural = smooth_batch(
                neural,
                std=model_args.dataset.data_transforms.smooth_kernel_std,
                kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
                device="cpu",
            )
            input_len = int(
                adjusted_lengths(
                    torch.tensor([n_time_steps]),
                    smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
                    smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
                    patch_size=model_args.model.patch_size,
                    patch_stride=model_args.model.patch_stride,
                )[0].item()
            )

            with torch.no_grad():
                baseline_logits = baseline(neural, day_idx)[0, :input_len, :].detach().numpy()
                proposed_logits = proposed(neural, day_idx)[0, :input_len, :].detach().numpy()

            baseline_ids = decode_argmax_ids(baseline_logits)
            proposed_ids = decode_argmax_ids(proposed_logits)
            baseline_partial, baseline_candidates = lm_decode(r, baseline_logits)
            proposed_partial, proposed_candidates = lm_decode(r, proposed_logits)

            baseline_best = baseline_candidates[0]
            proposed_best = proposed_candidates[0]
            changed = "si" if baseline_best != proposed_best else "no"

            baseline_per = proposed_per = baseline_wer = proposed_wer = None
            if true_ids is not None:
                baseline_per, baseline_wer = compute_metrics(baseline_ids, true_ids, baseline_best, sentence)
                proposed_per, proposed_wer = compute_metrics(proposed_ids, true_ids, proposed_best, sentence)

                totals["baseline_per_edits"] += edit_distance(baseline_ids, true_ids)
                totals["proposed_per_edits"] += edit_distance(proposed_ids, true_ids)
                totals["phoneme_len"] += len(true_ids)

                ref_words = normalize_words(sentence)
                totals["baseline_wer_edits"] += edit_distance(normalize_words(baseline_best), ref_words)
                totals["proposed_wer_edits"] += edit_distance(normalize_words(proposed_best), ref_words)
                totals["word_len"] += len(ref_words)

            print(
                f"| {trial_key} | {sentence or '-'} | {baseline_best} | {proposed_best} | "
                f"{format_metric(baseline_per)} | {format_metric(proposed_per)} | "
                f"{format_metric(baseline_wer)} | {format_metric(proposed_wer)} | {changed} |"
            )

            details.append(
                {
                    "trial": trial_key,
                    "real_sentence": sentence,
                    "real_phonemes": phoneme_text(true_ids) if true_ids is not None else None,
                    "baseline_phonemes": phoneme_text(baseline_ids),
                    "proposed_phonemes": phoneme_text(proposed_ids),
                    "baseline_partial": baseline_partial,
                    "proposed_partial": proposed_partial,
                    "baseline_candidates": baseline_candidates[:5],
                    "proposed_candidates": proposed_candidates[:5],
                }
            )

    if totals["phoneme_len"] > 0:
        print()
        print("Resumen con referencias reales")
        print(f"PER baseline agregado: {totals['baseline_per_edits'] / totals['phoneme_len']:.4f}")
        print(f"PER propuesto agregado: {totals['proposed_per_edits'] / totals['phoneme_len']:.4f}")
        print(f"WER baseline agregado: {totals['baseline_wer_edits'] / totals['word_len']:.4f}")
        print(f"WER propuesto agregado: {totals['proposed_wer_edits'] / totals['word_len']:.4f}")
    else:
        print()
        print("El split test no incluye referencias reales; no se calculan PER ni WER.")

    print("\nDetalle por ensayo")
    for item in details:
        print(f"\n{item['trial']}")
        if item["real_sentence"]:
            print(f"Frase real: {item['real_sentence']}")
            print(f"Fonemas reales: {item['real_phonemes']}")
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
