import argparse
from pathlib import Path
import sys

from omegaconf import OmegaConf


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[0]
MODEL_TRAINING_DIR = REPO_ROOT / "model_training"
sys.path.insert(0, str(THIS_DIR))
sys.path.insert(0, str(MODEL_TRAINING_DIR))

from datos import count_parameters  # noqa: E402
from modelo_gru_compacto import CompactGRUDecoder  # noqa: E402
from rnn_model import GRUDecoder  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Compara numero de parametros baseline vs modelo propuesto.")
    parser.add_argument("--args_path", default=str(REPO_ROOT / "data" / "t15_pretrained_rnn_baseline" / "checkpoint" / "args.yaml"))
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    n_days = len(model_args.dataset.sessions)

    baseline = GRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_units=model_args.model.n_units,
        n_days=n_days,
        n_classes=model_args.dataset.n_classes,
        rnn_dropout=model_args.model.rnn_dropout,
        input_dropout=model_args.model.input_network.input_layer_dropout,
        n_layers=model_args.model.n_layers,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )

    proposed = CompactGRUDecoder(
        neural_dim=model_args.model.n_input_features,
        n_days=n_days,
        n_classes=model_args.dataset.n_classes,
        patch_size=model_args.model.patch_size,
        patch_stride=model_args.model.patch_stride,
    )

    baseline_total, baseline_trainable = count_parameters(baseline)
    proposed_total, proposed_trainable = count_parameters(proposed)
    reduction = 100 * (1 - proposed_total / baseline_total)

    print("Comparacion de parametros")
    print(f"Baseline RNN-GRU:     {baseline_total:,} parametros ({baseline_trainable:,} entrenables)")
    print(f"Modelo propuesto:     {proposed_total:,} parametros ({proposed_trainable:,} entrenables)")
    print(f"Reduccion aproximada: {reduction:.2f} %")


if __name__ == "__main__":
    main()
