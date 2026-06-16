import argparse
from pathlib import Path
import sys

import h5py
import redis
import torch
from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from compare_trials_text import (  # noqa: E402
    build_baseline,
    build_proposed,
    compute_metrics,
    decode_argmax_ids,
    format_metric,
    lm_decode,
    load_baseline,
    phoneme_text,
)
from datos import adjusted_lengths, smooth_batch  # noqa: E402


def print_prediction_block(title, model_name, model_label, checkpoint_per, phonemes, best_text, partial, per, wer, candidates):
    print()
    print(title)
    print("-" * len(title))
    print(f"Modelo: {model_name}")
    print(f"Etiqueta visible: {model_label}")
    print(f"PER checkpoint: {checkpoint_per}")
    print(f"PER: {format_metric(per)}")
    print(f"WER: {format_metric(wer)}")
    print(f"Texto parcial: {partial}")
    print(f"Fonemas predichos: {phonemes}")
    print(f"Texto predicho: {best_text}")
    print("Notas: Prediccion generada localmente con Redis y el modelo de lenguaje 1-gram.")
    print()
    print(f"Candidatos para {model_name}")
    print("-" * (16 + len(model_name)))
    for idx, sentence in enumerate(candidates[:5], start=1):
        print(f"{idx}. {sentence}")


def main():
    parser = argparse.ArgumentParser(
        description="Ejecuta baseline y modelo propuesto y prepara los campos para el panel admin."
    )
    parser.add_argument(
        "--args_path",
        default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"),
    )
    parser.add_argument(
        "--baseline_checkpoint",
        default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"),
    )
    parser.add_argument(
        "--proposed_checkpoint",
        default=str(THIS_DIR / "salidas" / "baseline_adapter_logit" / "best_checkpoint.pt"),
    )
    parser.add_argument(
        "--data_dir",
        default=str(REPO_ROOT / "data" / "hdf5_data_final"),
    )
    parser.add_argument("--session", required=True)
    parser.add_argument("--split", choices=["val", "test"], required=True)
    parser.add_argument("--trial", required=True)
    parser.add_argument("--redis_host", default="localhost")
    parser.add_argument("--redis_port", type=int, default=6379)
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    sessions = list(model_args.dataset.sessions)
    if args.session not in sessions:
        raise ValueError(f"No existe la sesion {args.session} en args.yaml")
    session_index = sessions.index(args.session)

    baseline = load_baseline(build_baseline(model_args), args.baseline_checkpoint)

    proposed = build_proposed(model_args)
    proposed_checkpoint = torch.load(args.proposed_checkpoint, map_location="cpu", weights_only=False)
    proposed.load_state_dict(proposed_checkpoint["model_state_dict"])
    proposed.eval()

    r = redis.Redis(host=args.redis_host, port=args.redis_port, db=0)
    r.ping()

    hdf5_path = Path(args.data_dir) / args.session / f"data_{args.split}.hdf5"
    with h5py.File(hdf5_path, "r") as h5_file:
        if args.trial not in h5_file:
            available = ", ".join(list(h5_file.keys())[:10])
            raise KeyError(f"No existe {args.trial}. Primeros ensayos disponibles: {available}")

        trial = h5_file[args.trial]
        neural = torch.tensor(trial["input_features"][:], dtype=torch.float32).unsqueeze(0)
        n_time_steps = int(trial.attrs["n_time_steps"])
        day_idx = torch.tensor([session_index])
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

    baseline_per = proposed_per = baseline_wer = proposed_wer = None
    if true_ids is not None:
        baseline_per, baseline_wer = compute_metrics(baseline_ids, true_ids, baseline_best, sentence)
        proposed_per, proposed_wer = compute_metrics(proposed_ids, true_ids, proposed_best, sentence)

    print()
    print("Ensayo")
    print("------")
    print(f"Sesion: {args.session}")
    print(f"Particion: {args.split}")
    print(f"Trial: {args.trial}")
    if sentence:
        print(f"Frase real: {sentence}")
    else:
        print("Frase real: Sin frase disponible")

    print_prediction_block(
        "Prediccion baseline",
        "baseline",
        "Baseline RNN-GRU preentrenado",
        "0.1010",
        phoneme_text(baseline_ids),
        baseline_best,
        baseline_partial,
        baseline_per,
        baseline_wer,
        baseline_candidates,
    )
    print_prediction_block(
        "Prediccion modelo propuesto",
        "proposed",
        "Modelo propuesto con adaptador residual y corrector temporal",
        "0.0748",
        phoneme_text(proposed_ids),
        proposed_best,
        proposed_partial,
        proposed_per,
        proposed_wer,
        proposed_candidates,
    )
    print()
    print("Indicaciones")
    print("------------")
    print("1. En el panel admin, entra en editar el ensayo.")
    print("2. En Crear prediccion, selecciona el modelo en el desplegable.")
    print("3. Copia PER, WER, texto parcial, fonemas predichos, texto predicho y notas.")
    print("4. Guarda la prediccion.")
    print("5. En la prediccion creada, anade los candidatos con su rank.")
    if true_ids is None:
        print("Nota: este ensayo es de test; PER y WER aparecen como '-' porque no hay referencias reales.")


if __name__ == "__main__":
    main()
