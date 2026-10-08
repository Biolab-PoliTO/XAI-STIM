"""
XAI-STIM All-in-One Data Split

Organize processed patient folders into train, validation, and test sets.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

import shutil
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PROCESSED = ROOT / "dataset" / "processed"
SPLIT_TABLE = ROOT / "results" / "AiO_data_split.csv"
SPLITS = ("train", "val", "test")
DATASETS = ("LeadPoint", "NeuroSet", "AlphaMap")


def count_windows(folder):
    tables = [pd.read_csv(path, usecols=["class"]) for path in folder.glob("*.csv")]
    if not tables:
        raise FileNotFoundError(f"No window metadata found in {folder}")
    classes = pd.concat(tables, ignore_index=True)["class"]
    return int((classes == 0).sum()), int((classes == 1).sum())


def split_data():
    table = pd.read_csv(SPLIT_TABLE, sep=";", dtype={"patient": str})
    if table.duplicated(["patient", "dataset"]).any():
        raise ValueError("AiO_data_split.csv contains duplicated patient/dataset rows")
    if not set(table["split"]).issubset(SPLITS) or not set(table["dataset"]).issubset(DATASETS):
        raise ValueError("Unknown split or dataset in AiO_data_split.csv")

    moves, n0, n1 = [], [], []
    for row in table.itertuples(index=False):
        source = PROCESSED / row.dataset / row.patient
        destination = PROCESSED / row.split / row.dataset / row.patient
        if source.exists() and destination.exists():
            raise FileExistsError(f"Both source and destination exist for {row.dataset}/{row.patient}")
        folder = source if source.exists() else destination
        if not folder.exists():
            raise FileNotFoundError(f"Patient folder not found: {row.dataset}/{row.patient}")
        class_0, class_1 = count_windows(folder)
        n0.append(class_0)
        n1.append(class_1)
        if source.exists():
            moves.append((source, destination))

    for split in SPLITS:
        for dataset in DATASETS:
            (PROCESSED / split / dataset).mkdir(parents=True, exist_ok=True)
    for source, destination in moves:
        shutil.move(str(source), str(destination))
    for dataset in DATASETS:
        folder = PROCESSED / dataset
        if folder.exists() and not any(folder.iterdir()):
            folder.rmdir()

    table["n0"], table["n1"] = n0, n1
    table.to_csv(SPLIT_TABLE, sep=";", index=False)
    print(f"Moved {len(moves)} patients and updated {SPLIT_TABLE}")


if __name__ == "__main__":
    split_data()
