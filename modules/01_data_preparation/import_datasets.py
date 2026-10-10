"""
XAI-STIM Dataset Importer

Download LeadPoint, NeuroSet, and AlphaMap and assemble one zero-padded signal
matrix and one metadata table for each patient hemisphere.

Author: Fabrizio Sciscenti 
        H-MoVe, Department of Electronics and Telecommunications, Politecnico di Torino
        fabrizio.sciscenti@polito.it
Last updated: 10 October 2026
"""

import shutil
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import py7zr


# Open-data links
LEADPOINT_URL = "https://data.mendeley.com/public-api/zip/tr93krswn2/download/1"
NEUROSET_ALPHAMAP_URL = "https://zenodo.org/records/23259894/files/dataset_v2.zip?download=1"

ROOT = Path(__file__).resolve().parents[2]
DOWNLOADS = ROOT / "dataset" / ".downloads" 
OUTPUT = ROOT / "dataset" / "orig"
COLUMNS = ["patient", "side", "dataset", "depth", "electrode", "length", "class"]
SAMPLING_FREQUENCIES = {"LeadPoint": 24000, "NeuroSet": 20000, "AlphaMap": 12000}


def download(url, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        print(f"Downloading {url}")
        temporary = path.with_name(path.name + ".part")
        downloaded = temporary.stat().st_size if temporary.exists() else 0
        headers = {"User-Agent": "XAI-STIM dataset importer/1.0"}
        if downloaded:
            headers["Range"] = f"bytes={downloaded}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers)) as response:
            resume = downloaded > 0 and response.status == 206
            if downloaded and not resume:
                downloaded = 0
            total = downloaded + int(response.headers.get("Content-Length", 0))
            with temporary.open("ab" if resume else "wb") as output:
                while chunk := response.read(8 * 1024 * 1024):
                    output.write(chunk)
                    downloaded += len(chunk)
                    percent = 100 * downloaded / total if total else 0
                    print(
                        f"\r  {downloaded / 1024**2:.1f} / "
                        f"{total / 1024**2:.1f} MiB ({percent:.1f}%)",
                        end="",
                        flush=True,
                    )
            print()
        temporary.replace(path)
    return path


def metadata(table, dataset):
    table = table.copy()
    table.columns = table.columns.str.lower()
    table["patient"] = table["patient"].map(
        lambda value: f"P{int(''.join(filter(str.isdigit, str(value)))):03d}"
    )
    table["side"] = table["side"].astype(str).str.upper()
    table["dataset"] = dataset
    return table[COLUMNS]


def save(rows, table, dataset):
    patient, side = table.loc[0, ["patient", "side"]]
    folder = OUTPUT / dataset / patient
    folder.mkdir(parents=True, exist_ok=True)
    signals = np.zeros(
        (len(rows), max(map(len, rows))), np.result_type(*(row.dtype for row in rows))
    )
    for index, row in enumerate(rows):
        signals[index, : len(row)] = row
    np.savez_compressed(folder / f"{patient}_{side}.npz", signals=signals)
    (folder / f"{patient}_{side}.npy").unlink(missing_ok=True)
    table.to_csv(folder / f"{patient}_{side}.csv", index=False)
    print(f"  {dataset}/{patient}/{patient}_{side}: {signals.shape}")


def import_LeadPoint():
    """Import the LeadPoint train, validation, and test Feather/NPZ pairs."""
    cache = DOWNLOADS / "LeadPoint"
    outer = download(LEADPOINT_URL, cache / "LeadPoint.zip")
    source = cache / "raw_data"

    if not list(source.rglob("*.feather")):
        cache.mkdir(parents=True, exist_ok=True)
        nested = cache / "raw_data.7z"
        with zipfile.ZipFile(outer) as archive:
            member = next(name for name in archive.namelist() if name.endswith(".7z"))
            with archive.open(member) as compressed, nested.open("wb") as output:
                shutil.copyfileobj(compressed, output)
        source.mkdir(parents=True, exist_ok=True)
        with py7zr.SevenZipFile(nested) as archive:
            targets = [
                name
                for name in archive.getnames()
                if Path(name).suffix.lower() in {".feather", ".npz"}
            ]
            archive.extract(source, targets=targets)
        nested.unlink()

    feather = {path.stem: path for path in source.rglob("raw_*.feather")}
    npz = {path.stem: path for path in source.rglob("raw_*.npz")}
    stems = ["raw_all"] if "raw_all" in feather and "raw_all" in npz else sorted(feather.keys() & npz.keys())
    groups, arrays, temporary = {}, [], []

    for stem in stems:
        table = metadata(pd.read_feather(feather[stem]).reset_index(drop=True), "LeadPoint")
        table = table[table["electrode"].astype(str).str.strip().str.casefold() == "central"]
        matrix_path = cache / f"{stem}.npy"
        if not matrix_path.exists():
            with zipfile.ZipFile(npz[stem]) as archive:
                member = next(name for name in archive.namelist() if name.endswith(".npy"))
                with archive.open(member) as compressed, matrix_path.open("wb") as output:
                    shutil.copyfileobj(compressed, output)
        matrix = np.load(matrix_path, mmap_mode="r")
        arrays.append(matrix)
        temporary.append(matrix_path)
        for key, group in table.groupby(["patient", "side"]):
            groups.setdefault(key, []).append((matrix, group))

    for key in sorted(groups):
        rows, tables = [], []
        for matrix, group in groups[key]:
            rows += [matrix[index, : int(group.at[index, "length"])] for index in group.index]
            tables.append(group.reset_index(drop=True))
        save(rows, pd.concat(tables, ignore_index=True), "LeadPoint")

    for matrix in arrays:
        matrix._mmap.close()
    for path in temporary:
        path.unlink()


def import_npz_dataset(dataset):
    """Import one dataset from the shared NeuroSet/AlphaMap Zenodo archive."""
    cache = DOWNLOADS / "NeuroSet_AlphaMap"
    archive = download(NEUROSET_ALPHAMAP_URL, cache / "dataset_v2.zip")
    source = cache / "dataset_v2"
    marker = source / ".complete"
    if not marker.exists():
        source.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive) as compressed:
            compressed.extractall(source)
        marker.touch()

    source = next((path for path in source.rglob(dataset) if path.is_dir()), None)
    if source is None:
        raise FileNotFoundError(f"{dataset}/ not found in {archive}")

    for csv_path in sorted(source.rglob("*.csv")):
        npz_path = csv_path.with_suffix(".npz")
        if not npz_path.exists():
            continue
        table = metadata(pd.read_csv(csv_path, sep=None, engine="python"), dataset)
        with np.load(npz_path) as archive:
            matrix = archive[archive.files[0]]
            for _, group in table.groupby(["patient", "side"]):
                rows = [
                    matrix[index, : int(group.at[index, "length"])]
                    for index in group.index
                ]
                save(rows, group.reset_index(drop=True), dataset)


def import_NeuroSet():
    import_npz_dataset("NeuroSet")


def import_AlphaMap():
    import_npz_dataset("AlphaMap")


def compute_stats():
    """Compute dataset-level statistics from the assembled metadata CSV files."""
    statistics = {}
    for dataset, frequency in SAMPLING_FREQUENCIES.items():
        files = sorted((OUTPUT / dataset).rglob("*.csv"))
        if not files:
            continue
        table = pd.concat((pd.read_csv(path) for path in files), ignore_index=True)
        recordings = table.groupby(["patient", "side"]).size()
        inside = table[table["class"] == 1]["length"] / frequency
        outside = table[table["class"] != 1]["length"] / frequency
        depth_min, depth_max = table["depth"].min(), table["depth"].max()
        statistics[dataset] = [
            table["patient"].nunique(),
            f"{recordings.mean():.1f} ({recordings.std():.1f})",
            f"[{depth_min/1000:.1f} / {depth_max/1000:.1f}]",
            frequency / 1000,
            len(inside),
            f"{inside.mean():.1f} ({inside.std():.1f})",
            len(outside),
            f"{outside.mean():.1f} ({outside.std():.1f})",
        ]

    rows = [
        "Number of patients",
        "Mean (std) Recordings per hemisphere",
        "Depth range (mm) [min/max]",
        "Sampling frequency (kHz)",
        "Traces IN-STN",
        "Mean (std) duration of IN-STN traces (s)",
        "Traces OUT-STN",
        "Mean (std) duration of OUT-STN traces (s)",
    ]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame({"stat": rows, **statistics})
    result.to_csv(OUTPUT / "dataset_stats.csv", index=False)
    print(f"Statistics saved in {OUTPUT / 'dataset_stats.csv'}")


if __name__ == "__main__":
    import_LeadPoint()
    import_NeuroSet()
    import_AlphaMap()
    compute_stats()
    print(f"Dataset ready in {OUTPUT}")
