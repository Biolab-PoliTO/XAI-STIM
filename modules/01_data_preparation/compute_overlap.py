"""
XAI-STIM Overlap Computation

Compute patient/class overlaps that balance the number of one-second windows
first within each patient and then across datasets.

Author: Fabrizio Sciscenti
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 7 October 2026
"""

from pathlib import Path
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "dataset" / "orig"
OUTPUT = ROOT / "results" / "overlap.csv"
SAMPLING_FREQUENCIES = {"LeadPoint": 24000, "NeuroSet": 20000, "AlphaMap": 12000}
TARGET_FS = 10000 # Target sampling frequency for all datasets
WINDOW = 10000 # Number of samples in one-second windows (10 kHz * 1 s)
ETA_MIN = 0.25 # Minimum overlap 
ETA_MAX = 0.75 # Maximum overlap
ETA_STEP = 0.05 # Overlap step
TOLERANCE = 10 # Tolerance for overlap adjustment


def number_of_windows(length, eta):
    stride = np.round(WINDOW * (1 - np.asarray(eta)))
    return np.floor((np.asarray(length) - WINDOW) / stride).astype(int) + 1


def load_metadata():
    tables = []
    for dataset, frequency in SAMPLING_FREQUENCIES.items():
        for path in sorted((INPUT / dataset).rglob("*.csv")):
            table = pd.read_csv(path, usecols=lambda column: column in {"patient", "length", "class"})
            table["dataset"] = dataset
            table["length"] = np.floor(table["length"] * TARGET_FS / frequency).astype(int)
            table["class"] = (table["class"] == 1).astype(int)
            tables.append(table)
    if not tables:
        raise FileNotFoundError(f"No metadata CSV files found in {INPUT}")
    return pd.concat(tables, ignore_index=True)


def summarize(table, step):
    rows = []
    for (patient, dataset), group in table.groupby(["patient", "dataset"]):
        row = {"patient": patient, "dataset": dataset}
        for class_ in (0, 1):
            subset = group[group["class"] == class_]
            row[f"{step}_eta{class_}"] = round(subset["eta"].iloc[0], 2)
            row[f"{step}_n{class_}"] = int(subset["n"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def compute_overlap(table):
    table = table.copy()
    table["eta"] = ETA_MIN
    table["n"] = number_of_windows(table["length"], ETA_MIN)
    step0 = summarize(table, "step0")

    print("Step 1: balancing classes within each patient")
    for dataset, patient in table[["dataset", "patient"]].drop_duplicates().itertuples(index=False):
        patient_rows = (table["dataset"] == dataset) & (table["patient"] == patient)
        group = table.loc[patient_rows]
        lengths = [group.loc[group["class"] == class_, "length"].sum() for class_ in (0, 1)]
        counts = [group.loc[group["class"] == class_, "n"].sum() for class_ in (0, 1)]
        minority = int(np.argmin(counts))
        eta = np.round((1 - (lengths[minority] - WINDOW) / (WINDOW * counts[1 - minority])) / ETA_STEP) * ETA_STEP
        capped = eta > ETA_MAX
        eta = min(eta, ETA_MAX)
        rows = patient_rows & (table["class"] == minority)
        table.loc[rows, "eta"] = eta
        table.loc[rows, "n"] = number_of_windows(table.loc[rows, "length"], eta)

        if capped:
            eta = np.round((1 - (lengths[1 - minority] - WINDOW) / (WINDOW * counts[minority])) / ETA_STEP) * ETA_STEP
            eta = max(eta, 0)
            rows = patient_rows & (table["class"] == 1 - minority)
            table.loc[rows, "eta"] = eta
            table.loc[rows, "n"] = number_of_windows(table.loc[rows, "length"], eta)

    step1 = summarize(table, "step1")

    print("Step 2: balancing total windows across datasets")
    totals = table.groupby("dataset")["n"].sum()
    target = int(totals.mean())
    for dataset, total in totals.items():
        delta = target - total
        patients = table.loc[table["dataset"] == dataset, "patient"].unique()
        counts = np.array([
            table.loc[(table["dataset"] == dataset) & (table["patient"] == patient), "n"].sum()
            for patient in patients
        ])
        targets = counts + (counts / counts.sum() * delta).astype(int)

        for patient, patient_target in zip(patients, targets):
            rows = (table["dataset"] == dataset) & (table["patient"] == patient)
            lengths = table.loc[rows, "length"].to_numpy()
            eta = table.loc[rows, "eta"].to_numpy(copy=True)
            count = table.loc[rows, "n"].sum()
            while abs(count - patient_target) > TOLERANCE:
                eta = eta + np.sign(delta) * ETA_STEP
                if (eta <= ETA_MAX).all() and (eta >= 0).all():
                    count = number_of_windows(lengths, eta).sum()
                else:
                    eta = np.clip(eta, 0, ETA_MAX)
                    count = number_of_windows(lengths, eta).sum()
                    break
            table.loc[rows, "eta"] = eta
            table.loc[rows, "n"] = number_of_windows(lengths, eta)

    result = step0.merge(step1, on=["patient", "dataset"]).merge(
        summarize(table, "step2"), on=["patient", "dataset"]
    )
    columns = [
        "patient", "dataset", "step0_eta0", "step0_eta1", "step0_n0", "step0_n1",
        "step1_eta0", "step1_eta1", "step1_n0", "step1_n1",
        "step2_eta0", "step2_eta1", "step2_n0", "step2_n1",
    ]
    return result[columns]


if __name__ == "__main__":
    result = compute_overlap(load_metadata())
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUTPUT, index=False)
    print(f"Saved {len(result)} patient rows in {OUTPUT}")
