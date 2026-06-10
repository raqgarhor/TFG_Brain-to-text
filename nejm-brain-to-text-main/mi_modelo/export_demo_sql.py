import argparse
from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np
import redis
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
PROJECT_ROOT = REPO_ROOT.parent
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def sql_literal(value):
    if value is None:
        return "null"
    if isinstance(value, (int, float, np.integer, np.floating)):
        return str(value)
    text = str(value)
    return "'" + text.replace("'", "''") + "'"


def sql_numeric(value):
    if value is None:
        return "null"
    return f"{float(value):.6f}"


def normalize_words(text):
    if not text:
        return []
    cleaned = "".join(ch.lower() if ch.isalnum() or ch.isspace() else " " for ch in text)
    return [word for word in cleaned.split() if word]


def phoneme_text(ids):
    return " ".join(ids_to_phonemes(ids))


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
    return checkpoint


def load_proposed(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return checkpoint


def lm_decode(redis_client, logits):
    redis_client.flushall()
    input_stream = "remote_lm_input"
    partial_stream = "remote_lm_output_partial"
    final_stream = "remote_lm_output_final"

    reset_seen = get_current_redis_time_ms(redis_client)
    partial_seen = get_current_redis_time_ms(redis_client)
    final_seen = get_current_redis_time_ms(redis_client)

    reset_seen = reset_remote_language_model(redis_client, reset_seen)
    lm_logits = rearrange_speech_logits_pt(logits[None, :, :])[0]
    partial_seen, partial_text = send_logits_to_remote_lm(
        redis_client,
        input_stream,
        partial_stream,
        partial_seen,
        lm_logits,
    )
    final_seen, lm_out = finalize_remote_lm(redis_client, final_stream, final_seen)
    candidates = [str(item) for item in lm_out["candidate_sentences"]]
    return str(partial_text), candidates


def collect_trials(data_dir, sessions, split, limit):
    items = []
    for session_index, session in enumerate(sessions):
        hdf5_path = Path(data_dir) / session / f"data_{split}.hdf5"
        if not hdf5_path.exists():
            continue
        with h5py.File(hdf5_path, "r") as h5_file:
            for trial_key in list(h5_file.keys()):
                items.append(
                    {
                        "session": session,
                        "session_index": session_index,
                        "split": split,
                        "trial_key": trial_key,
                        "hdf5_path": hdf5_path,
                    }
                )
                if len(items) >= limit:
                    return items
    return items


def prediction_sql(trial_cte, prediction):
    candidates = prediction["candidates"][:5]
    values = ",\n        ".join(
        f"({rank}, {sql_literal(sentence)})"
        for rank, sentence in enumerate(candidates, start=1)
    )
    if not values:
        values = "(1, '')"

    return f"""
with trial_row as (
    select id from trials
    where session_name = {sql_literal(trial_cte["session_name"])}
      and split = {sql_literal(trial_cte["split"])}
      and trial_key = {sql_literal(trial_cte["trial_key"])}
),
prediction_row as (
    insert into predictions (
        trial_id,
        model_name,
        model_label,
        predicted_phonemes,
        predicted_text,
        partial_text,
        per_value,
        wer_value,
        checkpoint_per,
        notes
    )
    select
        id,
        {sql_literal(prediction["model_name"])},
        {sql_literal(prediction["model_label"])},
        {sql_literal(prediction["predicted_phonemes"])},
        {sql_literal(prediction["predicted_text"])},
        {sql_literal(prediction["partial_text"])},
        {sql_numeric(prediction["per_value"])},
        {sql_numeric(prediction["wer_value"])},
        {sql_numeric(prediction["checkpoint_per"])},
        {sql_literal(prediction["notes"])}
    from trial_row
    on conflict (trial_id, model_name) do update set
        model_label = excluded.model_label,
        predicted_phonemes = excluded.predicted_phonemes,
        predicted_text = excluded.predicted_text,
        partial_text = excluded.partial_text,
        per_value = excluded.per_value,
        wer_value = excluded.wer_value,
        checkpoint_per = excluded.checkpoint_per,
        notes = excluded.notes,
        updated_at = now()
    returning id
),
deleted_candidates as (
    delete from candidates
    where prediction_id in (select id from prediction_row)
)
insert into candidates (prediction_id, rank, candidate_text)
select prediction_row.id, candidate_data.rank, candidate_data.candidate_text
from prediction_row
cross join (
    values
        {values}
) as candidate_data(rank, candidate_text)
on conflict (prediction_id, rank) do update set
    candidate_text = excluded.candidate_text;
"""


def trial_sql(trial):
    return f"""
insert into trials (
    session_name,
    split,
    trial_key,
    real_sentence,
    real_phonemes,
    input_shape,
    logits_shape,
    signal_image_path,
    notes
)
values (
    {sql_literal(trial["session_name"])},
    {sql_literal(trial["split"])},
    {sql_literal(trial["trial_key"])},
    {sql_literal(trial["real_sentence"])},
    {sql_literal(trial["real_phonemes"])},
    {sql_literal(trial["input_shape"])},
    {sql_literal(trial["logits_shape"])},
    {sql_literal(trial["signal_image_path"])},
    {sql_literal(trial["notes"])}
)
on conflict (session_name, split, trial_key) do update set
    real_sentence = excluded.real_sentence,
    real_phonemes = excluded.real_phonemes,
    input_shape = excluded.input_shape,
    logits_shape = excluded.logits_shape,
    signal_image_path = excluded.signal_image_path,
    notes = excluded.notes,
    updated_at = now();
"""


def save_signal_image(features, smoothed, item, output_dir, feature_index):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original = features[0, :, feature_index].detach().cpu().numpy()
    smooth = smoothed[0, :, feature_index].detach().cpu().numpy()
    filename = f"{item['session']}_{item['split']}_{item['trial_key']}_feature_{feature_index}.png"
    file_path = output_dir / filename

    plt.figure(figsize=(10, 4), dpi=120)
    plt.plot(original, color="#2b6cb0", linewidth=0.8, alpha=0.45, label="Senal original")
    plt.plot(
        np.arange(len(smooth)),
        smooth,
        color="#d97706",
        linewidth=1.8,
        label="Suavizado gaussiano",
    )
    plt.title(f"{item['session']} - {item['trial_key']} - caracteristica {feature_index}")
    plt.xlabel("Tiempo")
    plt.ylabel("Valor de la caracteristica neuronal")
    plt.grid(True, alpha=0.25)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()

    return f"/images/signals/{filename}"


def process_trial(item, model_args, baseline, proposed, redis_client, checkpoint_per, image_dir, feature_index):
    with h5py.File(item["hdf5_path"], "r") as h5_file:
        trial = h5_file[item["trial_key"]]
        features = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        n_time_steps = int(trial.attrs["n_time_steps"])
        sentence = trial.attrs.get("sentence_label")

        true_ids = None
        if "seq_class_ids" in trial:
            seq_len = int(trial.attrs["seq_len"])
            true_ids = [int(x) for x in trial["seq_class_ids"][:seq_len]]

    day_idx = torch.tensor([item["session_index"]])
    smoothed = smooth_batch(
        features,
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
        baseline_logits = baseline(smoothed, day_idx)[0, :input_len, :].detach().numpy()
        proposed_logits = proposed(smoothed, day_idx)[0, :input_len, :].detach().numpy()

    baseline_ids = decode_argmax_ids(baseline_logits)
    proposed_ids = decode_argmax_ids(proposed_logits)
    baseline_partial, baseline_candidates = lm_decode(redis_client, baseline_logits)
    proposed_partial, proposed_candidates = lm_decode(redis_client, proposed_logits)
    signal_image_path = save_signal_image(features, smoothed, item, image_dir, feature_index)

    trial_info = {
        "session_name": item["session"],
        "split": item["split"],
        "trial_key": item["trial_key"],
        "real_sentence": sentence,
        "real_phonemes": phoneme_text(true_ids) if true_ids is not None else None,
        "input_shape": str(tuple(smoothed.shape)),
        "logits_shape": str(tuple(baseline_logits.shape)),
        "signal_image_path": signal_image_path,
        "notes": "Ensayo generado localmente para la aplicacion demostradora.",
    }

    predictions = []
    for model_name, model_label, pred_ids, partial, candidates, per_ckpt in [
        ("baseline", "Baseline RNN-GRU preentrenado", baseline_ids, baseline_partial, baseline_candidates, checkpoint_per["baseline"]),
        ("proposed", "Modelo propuesto con adaptador residual y corrector temporal", proposed_ids, proposed_partial, proposed_candidates, checkpoint_per["proposed"]),
    ]:
        per_value = None
        if true_ids is not None and len(true_ids) > 0:
            per_value = edit_distance(pred_ids, true_ids) / len(true_ids)

        wer_value = None
        if sentence and candidates:
            ref_words = normalize_words(sentence)
            hyp_words = normalize_words(candidates[0])
            if ref_words:
                wer_value = edit_distance(hyp_words, ref_words) / len(ref_words)

        predictions.append(
            {
                "model_name": model_name,
                "model_label": model_label,
                "predicted_phonemes": phoneme_text(pred_ids),
                "predicted_text": candidates[0] if candidates else None,
                "partial_text": partial,
                "candidates": candidates,
                "per_value": per_value,
                "wer_value": wer_value,
                "checkpoint_per": per_ckpt,
                "notes": "Prediccion generada localmente con Redis y el modelo de lenguaje 1-gram.",
            }
        )

    return trial_info, predictions


def main():
    parser = argparse.ArgumentParser(
        description="Genera un SQL con ensayos y predicciones demo para la aplicacion Spring Boot."
    )
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--baseline_checkpoint", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"))
    parser.add_argument("--proposed_checkpoint", default=str(THIS_DIR / "salidas" / "baseline_adapter_logit" / "best_checkpoint.pt"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "brain-to-text-web" / "database" / "002_seed_demo_results.sql"))
    parser.add_argument("--image_dir", default=str(PROJECT_ROOT / "brain-to-text-web" / "src" / "main" / "resources" / "static" / "images" / "signals"))
    parser.add_argument("--feature_index", type=int, default=0)
    parser.add_argument("--val_count", type=int, default=50)
    parser.add_argument("--test_count", type=int, default=50)
    parser.add_argument("--candidate_count", type=int, default=5)
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)

    baseline = build_baseline(model_args)
    baseline_checkpoint = load_baseline(baseline, args.baseline_checkpoint)

    proposed = build_proposed(model_args)
    proposed_checkpoint = load_proposed(proposed, args.proposed_checkpoint)

    checkpoint_per = {
        "baseline": baseline_checkpoint.get("val_PER"),
        "proposed": proposed_checkpoint.get("val_PER"),
    }

    items = []
    items.extend(collect_trials(args.data_dir, sessions, "val", args.val_count))
    items.extend(collect_trials(args.data_dir, sessions, "test", args.test_count))

    expected = args.val_count + args.test_count
    if len(items) < expected:
        print(f"Aviso: se solicitaron {expected} ensayos, pero solo se encontraron {len(items)}.")

    redis_client = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    redis_client.ping()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    statements = [
        "-- Datos demo generados localmente para la aplicacion Brain-to-Text.",
        "-- Ejecutar despues de 001_initial_schema.sql.",
        "begin;",
    ]

    for index, item in enumerate(items, start=1):
        print(f"[{index}/{len(items)}] {item['split']} {item['session']} {item['trial_key']}")
        trial_info, predictions = process_trial(
            item,
            model_args,
            baseline,
            proposed,
            redis_client,
            checkpoint_per,
            args.image_dir,
            args.feature_index,
        )
        statements.append(trial_sql(trial_info))
        for prediction in predictions:
            prediction["candidates"] = prediction["candidates"][: args.candidate_count]
            statements.append(prediction_sql(trial_info, prediction))

    statements.append("commit;")
    output_path.write_text("\n".join(statements), encoding="utf-8")
    print()
    print(f"SQL generado: {output_path}")
    print(f"Ensayos incluidos: {len(items)}")
    print(f"Predicciones incluidas: {len(items) * 2}")


if __name__ == "__main__":
    main()
