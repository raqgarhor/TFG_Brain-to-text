import argparse
import os
import re

import editdistance
import h5py
import pandas as pd
from omegaconf import OmegaConf


def clean_sentence(sentence):
    sentence = str(sentence)
    sentence = re.sub(r"[^a-zA-Z\- \']", "", sentence)
    sentence = sentence.replace("- ", " ").lower()
    sentence = sentence.replace("--", "").lower()
    sentence = sentence.replace(" '", "'").lower()
    sentence = sentence.strip()
    return " ".join(word for word in sentence.split() if word)


def main():
    parser = argparse.ArgumentParser(
        description="Calcula WER en validacion comparando un CSV de predicciones con los HDF5."
    )
    parser.add_argument(
        "--pred_csv",
        default="model_training/rnn_baseline_submission_file_valsplit.csv",
        help="CSV con columnas id,text.",
    )
    parser.add_argument(
        "--data_dir",
        default="data/hdf5_data_final",
        help="Carpeta hdf5_data_final.",
    )
    parser.add_argument(
        "--args_path",
        default="data/t15_pretrained_rnn_baseline/checkpoint/args.yaml",
        help="args.yaml del modelo, usado para respetar el orden de sesiones.",
    )
    args = parser.parse_args()

    model_args = OmegaConf.load(args.args_path)
    true_sentences = []
    trial_ids = []

    for session in model_args.dataset.sessions:
        hdf5_path = os.path.join(args.data_dir, session, "data_val.hdf5")
        if not os.path.exists(hdf5_path):
            continue

        with h5py.File(hdf5_path, "r") as h5_file:
            for trial_key in list(h5_file.keys()):
                label = h5_file[trial_key].attrs["sentence_label"]
                if isinstance(label, bytes):
                    label = label.decode()

                true_sentences.append(clean_sentence(label))
                trial_ids.append((session, trial_key))

    pred_sentences = (
        pd.read_csv(args.pred_csv)["text"].fillna("").map(clean_sentence).tolist()
    )

    if len(true_sentences) != len(pred_sentences):
        raise ValueError(
            "No coincide el numero de frases reales "
            f"({len(true_sentences)}) y predicciones ({len(pred_sentences)})."
        )

    total_words = 0
    total_edit_distance = 0
    per_trial = []

    for idx, (true_sentence, pred_sentence) in enumerate(
        zip(true_sentences, pred_sentences)
    ):
        true_words = true_sentence.split()
        pred_words = pred_sentence.split()
        edit_distance = editdistance.eval(true_words, pred_words)

        total_edit_distance += edit_distance
        total_words += len(true_words)
        per_trial.append(
            (
                edit_distance / max(1, len(true_words)),
                edit_distance,
                len(true_words),
                idx,
                true_sentence,
                pred_sentence,
                trial_ids[idx],
            )
        )

    print(f"Frases evaluadas: {len(true_sentences)}")
    print(f"Edit distance total: {total_edit_distance}")
    print(f"Palabras reales totales: {total_words}")
    print(f"WER agregado: {100 * total_edit_distance / total_words:.2f}%")

    print("\nPeores 5 ejemplos:")
    for rate, edit_distance, n_words, idx, true_sentence, pred_sentence, trial_id in sorted(
        per_trial, reverse=True
    )[:5]:
        session, trial_key = trial_id
        print(
            f"#{idx} {session} {trial_key} "
            f"WER={100 * rate:.1f}% ({edit_distance}/{n_words})"
        )
        print(f"  real:      {true_sentence}")
        print(f"  predicha:  {pred_sentence}")


if __name__ == "__main__":
    main()
