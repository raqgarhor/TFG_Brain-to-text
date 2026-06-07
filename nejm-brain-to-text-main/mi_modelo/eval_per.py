import argparse
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
sys.path.insert(0, str(THIS_DIR))

from datos import (  # noqa: E402
    HDF5TrialDataset,
    adjusted_lengths,
    collate_trials,
    decode_argmax,
    edit_distance,
    ids_to_phonemes,
    parse_sessions,
    smooth_batch,
)
from modelo_gru_compacto import CompactGRUDecoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Evalua PER del modelo GRU compacto.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--checkpoint", default=str(THIS_DIR / "salidas" / "gru_compacto" / "best_checkpoint.pt"))
    parser.add_argument("--sessions", default="t15.2023.08.13")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_val_trials", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--show_examples", type=int, default=3)
    parser.add_argument("--temporal_conv", action="store_true", help="Evaluar la variante con convolucion temporal.")
    parser.add_argument("--recurrent_type", choices=["gru", "lstm"], default=None, help="Tipo de capa recurrente. Si se omite, se intenta leer del checkpoint.")
    args = parser.parse_args()

    device = torch.device("cpu")
    model_args = OmegaConf.load(args.args_path)
    all_sessions = list(model_args.dataset.sessions)
    sessions = parse_sessions(all_sessions, args.sessions, args.max_sessions)
    session_to_day = {session: all_sessions.index(session) for session in all_sessions}

    dataset = HDF5TrialDataset(
        args.data_dir,
        sessions=sessions,
        split="val",
        session_to_day=session_to_day,
        max_trials_per_session=args.max_val_trials,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials, num_workers=0)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    recurrent_type = args.recurrent_type or checkpoint.get("recurrent_type", "gru")
    temporal_conv = args.temporal_conv or bool(checkpoint.get("temporal_conv", False))

    model = CompactGRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_days=len(all_sessions),
        n_classes=model_args.dataset.n_classes,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
        temporal_conv=temporal_conv,
        recurrent_type=recurrent_type,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    total_edits = 0
    total_len = 0
    examples_printed = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            seq_lens = batch["seq_lens"].to(device)
            day_idx = batch["day_idx"].to(device)
            n_time_steps = batch["n_time_steps"].to(device)

            features = smooth_batch(
                features,
                std=model_args.dataset.data_transforms.smooth_kernel_std,
                kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
                device=device,
            )
            logits = model(features, day_idx)
            input_lens = adjusted_lengths(
                n_time_steps.cpu(),
                smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
                smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
                patch_size=model_args.model.patch_size,
                patch_stride=model_args.model.patch_stride,
            )

            for i in range(logits.shape[0]):
                pred = decode_argmax(logits[i, : input_lens[i], :])
                true = labels[i, : seq_lens[i]].detach().cpu().tolist()

                total_edits += edit_distance(pred, true)
                total_len += len(true)

                if examples_printed < args.show_examples:
                    print(f"\nEjemplo {examples_printed + 1}")
                    print(f"Trial: {batch['trial_key'][i]}")
                    print(f"Fonemas predichos: {' '.join(ids_to_phonemes(pred))}")
                    print(f"Fonemas reales:    {' '.join(ids_to_phonemes(true))}")
                    examples_printed += 1

    per = total_edits / max(total_len, 1)
    print("\nEvaluacion del modelo propuesto")
    print(f"Sesiones: {', '.join(sessions)}")
    print(f"Ensayos evaluados: {len(dataset)}")
    print(f"Distancia total: {total_edits}")
    print(f"Longitud total de referencia: {total_len}")
    print(f"PER: {per:.4f} ({per * 100:.2f} %)")


if __name__ == "__main__":
    main()
