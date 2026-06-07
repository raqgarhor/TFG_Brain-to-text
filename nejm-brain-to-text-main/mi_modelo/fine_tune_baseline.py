import argparse
from itertools import cycle
from pathlib import Path
import sys

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from datos import (  # noqa: E402
    HDF5TrialDataset,
    adjusted_lengths,
    collate_trials,
    parse_sessions,
    smooth_batch,
)
from rnn_model import GRUDecoder  # noqa: E402


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


def load_baseline(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = {
        key.replace("module.", "").replace("_orig_mod.", ""): value
        for key, value in checkpoint["model_state_dict"].items()
    }
    model.load_state_dict(state_dict)
    return checkpoint


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


def set_trainable(model, mode):
    for param in model.parameters():
        param.requires_grad = False

    if mode == "output":
        for param in model.out.parameters():
            param.requires_grad = True
    elif mode == "day_output":
        for param in model.day_weights.parameters():
            param.requires_grad = True
        for param in model.day_biases.parameters():
            param.requires_grad = True
        for param in model.out.parameters():
            param.requires_grad = True
    elif mode == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("--trainable debe ser output, day_output o all")


def add_training_noise(features, noise_std, offset_std):
    if noise_std > 0:
        features = features + torch.randn_like(features) * noise_std
    if offset_std > 0:
        features = features + torch.randn(
            features.shape[0],
            1,
            features.shape[2],
            device=features.device,
        ) * offset_std
    return features


def main():
    parser = argparse.ArgumentParser(description="Fine-tuning reducido del baseline preentrenado.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    parser.add_argument("--checkpoint", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "best_checkpoint"))
    parser.add_argument("--data_dir", default=str(REPO_ROOT / "data" / "hdf5_data_final"))
    parser.add_argument("--sessions", default="t15.2023.08.13")
    parser.add_argument("--max_sessions", type=int, default=None)
    parser.add_argument("--max_train_trials", type=int, default=348)
    parser.add_argument("--max_val_trials", type=int, default=35)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_batches", type=int, default=300)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--lr_min", type=float, default=1e-6)
    parser.add_argument("--eval_every", type=int, default=50)
    parser.add_argument("--trainable", choices=["output", "day_output", "all"], default="day_output")
    parser.add_argument("--noise_std", type=float, default=0.0)
    parser.add_argument("--offset_std", type=float, default=0.0)
    parser.add_argument("--output_dir", default=str(THIS_DIR / "salidas" / "baseline_finetune"))
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
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_trials, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_trials, num_workers=0)

    model = build_baseline(model_args).to(device)
    checkpoint = load_baseline(model, args.checkpoint)
    set_trainable(model, args.trainable)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    initial_per = evaluate_per(model, val_loader, model_args, device)
    best_per = initial_per

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.num_batches, 1),
        eta_min=args.lr_min,
    )
    ctc_loss = torch.nn.CTCLoss(blank=0, reduction="mean", zero_infinity=True)

    print("Fine-tuning reducido del baseline preentrenado")
    print(f"Sesiones: {', '.join(sessions)}")
    print(f"Ensayos train: {len(train_dataset)}")
    print(f"Ensayos val: {len(val_dataset)}")
    print(f"Parametros totales: {total_params:,}")
    print(f"Parametros entrenables: {trainable_params:,}")
    print(f"Partes entrenables: {args.trainable}")
    print(f"PER checkpoint original en esta validacion: {initial_per:.4f}")
    if "val_PER" in checkpoint:
        print(f"PER global guardado por autores: {checkpoint['val_PER']:.4f}")

    train_iter = cycle(train_loader)
    for step in range(1, args.num_batches + 1):
        model.train()
        batch = next(train_iter)
        features = batch["features"].to(device)
        labels = batch["labels"].to(device)
        day_idx = batch["day_idx"].to(device)
        seq_lens = batch["seq_lens"].to(device)
        n_time_steps = batch["n_time_steps"].to(device)

        features = add_training_noise(features, args.noise_std, args.offset_std)
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
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        optimizer.step()
        scheduler.step()

        print(f"Batch {step:04d}/{args.num_batches} - CTC loss: {loss.item():.4f} - lr: {optimizer.param_groups[0]['lr']:.7f}")

        if step == args.num_batches or step % args.eval_every == 0:
            per = evaluate_per(model, val_loader, model_args, device)
            print(f"  PER validacion fine-tuning: {per:.4f}")
            if per < best_per:
                best_per = per
                output_dir = Path(args.output_dir)
                output_dir.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "val_PER": best_per,
                        "base_checkpoint": str(args.checkpoint),
                        "sessions": sessions,
                        "trainable": args.trainable,
                    },
                    output_dir / "best_checkpoint.pt",
                )
                print(f"  Checkpoint guardado: {output_dir / 'best_checkpoint.pt'}")

    print(f"Mejor PER fine-tuning: {best_per:.4f}")


if __name__ == "__main__":
    main()
