import argparse
from itertools import cycle
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
    count_parameters,
    parse_sessions,
    smooth_batch,
)
from modelo_gru_compacto import CompactGRUDecoder  # noqa: E402


def build_model(model_args, temporal_conv=False, recurrent_type="gru"):
    return CompactGRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_days=len(model_args.dataset.sessions),
        n_classes=model_args.dataset.n_classes,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
        projection_dim=512,
        projection_dropout=0.2,
        temporal_conv=temporal_conv,
        input_dropout=0.2,
        gru_units=256,
        gru_layers=2,
        gru_dropout=0.3,
        recurrent_type=recurrent_type,
    )


def evaluate_per(model, loader, model_args, device):
    model.eval()
    total_edits = 0
    total_len = 0

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            day_idx = batch["day_idx"].to(device)
            seq_lens = batch["seq_lens"].to(device)
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
                pred = torch.argmax(logits[i, : input_lens[i], :], dim=-1)
                pred = torch.unique_consecutive(pred).detach().cpu().tolist()
                pred = [p for p in pred if p != 0]
                true = labels[i, : seq_lens[i]].detach().cpu().tolist()
                total_edits += levenshtein(pred, true)
                total_len += len(true)

    return total_edits / max(total_len, 1)


def levenshtein(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (0 if ca == cb else 1))
            prev = old
    return dp[-1]


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento reducido del modelo GRU compacto.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--sessions", default="t15.2023.08.13", help="Sesiones separadas por comas.")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_train_trials", type=int, default=64)
    parser.add_argument("--max_val_trials", type=int, default=32)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--resume_checkpoint", default=None, help="Checkpoint desde el que continuar entrenando.")
    parser.add_argument("--partial_resume", action="store_true", help="Cargar solo los pesos compatibles del checkpoint.")
    parser.add_argument("--temporal_conv", action="store_true", help="Activar bloque convolucional temporal antes de la GRU.")
    parser.add_argument("--recurrent_type", choices=["gru", "lstm"], default="gru", help="Tipo de capa recurrente compacta.")
    parser.add_argument("--reset_best_per", action="store_true", help="No reutilizar el mejor PER guardado al cambiar el conjunto de validacion.")
    parser.add_argument("--eval_every", type=int, default=None, help="Cada cuantos batches evaluar PER.")
    parser.add_argument("--output_dir", default=str(THIS_DIR / "salidas" / "gru_compacto"))
    args = parser.parse_args()

    torch.manual_seed(10)
    device = torch.device("cpu")
    model_args = OmegaConf.load(args.args_path)
    all_sessions = list(model_args.dataset.sessions)
    sessions = parse_sessions(all_sessions, args.sessions, args.max_sessions)
    session_to_day = {session: all_sessions.index(session) for session in all_sessions}

    train_dataset = HDF5TrialDataset(
        args.data_dir,
        sessions=sessions,
        split="train",
        session_to_day=session_to_day,
        max_trials_per_session=args.max_train_trials,
    )
    val_dataset = HDF5TrialDataset(
        args.data_dir,
        sessions=sessions,
        split="val",
        session_to_day=session_to_day,
        max_trials_per_session=args.max_val_trials,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_trials,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_trials,
        num_workers=0,
    )

    model = build_model(
        model_args,
        temporal_conv=args.temporal_conv,
        recurrent_type=args.recurrent_type,
    ).to(device)
    total_params, trainable_params = count_parameters(model)

    if args.resume_checkpoint is not None:
        checkpoint_path = Path(args.resume_checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"No existe el checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        if args.partial_resume:
            current_state = model.state_dict()
            compatible_state = {
                key: value
                for key, value in checkpoint["model_state_dict"].items()
                if key in current_state and current_state[key].shape == value.shape
            }
            skipped_keys = [
                key
                for key, value in checkpoint["model_state_dict"].items()
                if key not in current_state or current_state[key].shape != value.shape
            ]
            load_result = model.load_state_dict(compatible_state, strict=False)
            print("Carga parcial activada.")
            if load_result.missing_keys:
                print("Pesos nuevos inicializados:", ", ".join(load_result.missing_keys))
            if skipped_keys:
                print("Pesos ignorados por incompatibilidad:", ", ".join(skipped_keys))
        else:
            model.load_state_dict(checkpoint["model_state_dict"])
        print(f"Checkpoint cargado: {checkpoint_path}")
        if "best_val_per" in checkpoint:
            print(f"PER guardado en checkpoint: {checkpoint['best_val_per']:.4f}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.num_batches, 1),
        eta_min=args.lr_min,
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    print("Entrenamiento reducido del modelo propuesto")
    print(f"Sesiones: {', '.join(sessions)}")
    print(f"Ensayos train: {len(train_dataset)}")
    print(f"Ensayos val: {len(val_dataset)}")
    print(f"Parametros totales: {total_params:,}")
    print(f"Parametros entrenables: {trainable_params:,}")
    print(f"Convolucion temporal: {'si' if args.temporal_conv else 'no'}")
    print(f"Capa recurrente: {args.recurrent_type.upper()}")

    best_per = float("inf")
    if args.resume_checkpoint is not None and "best_val_per" in checkpoint and not args.reset_best_per:
        best_per = float(checkpoint["best_val_per"])
    elif args.resume_checkpoint is not None and args.reset_best_per:
        print("Mejor PER reiniciado para esta configuracion de validacion.")

    train_iter = cycle(train_loader)
    eval_every = args.eval_every or max(5, args.num_batches // 4)

    for step in range(1, args.num_batches + 1):
        model.train()
        batch = next(train_iter)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        day_idx = batch["day_idx"].to(device)
        seq_lens = batch["seq_lens"].to(device)
        n_time_steps = batch["n_time_steps"].to(device)

        features = smooth_batch(
            features,
            std=model_args.dataset.data_transforms.smooth_kernel_std,
            kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            device=device,
        )
        logits = model(features, day_idx)
        input_lens = adjusted_lengths(
            n_time_steps,
            smooth_kernel_size=model_args.dataset.data_transforms.smooth_kernel_size,
            smooth_kernel_std=model_args.dataset.data_transforms.smooth_kernel_std,
            patch_size=model_args.model.patch_size,
            patch_stride=model_args.model.patch_stride,
        ).to(device)

        loss = ctc_loss(
            logits.log_softmax(2).permute(1, 0, 2),
            labels,
            input_lens,
            seq_lens,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Batch {step:04d}/{args.num_batches} - CTC loss: {loss.item():.4f} - lr: {current_lr:.6f}")

        if step == args.num_batches or step % eval_every == 0:
            per = evaluate_per(model, val_loader, model_args, device)
            print(f"  PER validacion reducido: {per:.4f}")
            if per < best_per:
                best_per = per
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "model_name": "CompactGRUDecoder",
                        "sessions": sessions,
                        "temporal_conv": args.temporal_conv,
                        "recurrent_type": args.recurrent_type,
                        "best_val_per": best_per,
                    },
                    output_dir / "best_checkpoint.pt",
                )
                print(f"  Checkpoint guardado: {output_dir / 'best_checkpoint.pt'}")

    print(f"Mejor PER validacion reducido: {best_per:.4f}")


if __name__ == "__main__":
    main()
