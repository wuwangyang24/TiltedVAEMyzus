"""
Simple dataset counter for:
1) Chemical-class selected compounds (classes with >= min compounds)
2) Compounds listed in 500ppm CSV

Examples:
python TiltedVAEMyzus/Tests/count_dataset_stats.py --chemical_metadata METADATA/metadata_compound_all100ppm.json --class_metadata METADATA/synthesisprogram_compoundno.csv --ppm500_csv TiltedVAEMyzus/Tests/efficacy500_classifier/compounds500ppm.csv --class_compound_col compound --label_col synthesis_program --min_compounds_per_class 30
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count total images for (1) class-filtered chemical-class compounds "
            "and (2) compounds listed in 500ppm CSV."
        )
    )
    parser.add_argument(
        "--chemical_metadata",
        required=True,
        help="Path to chemical-class metadata JSON (compound -> plate -> treated/control paths).",
    )
    parser.add_argument(
        "--class_metadata",
        required=True,
        help="CSV/XLSX metadata with compound and class label columns used for class filtering.",
    )
    parser.add_argument(
        "--ppm500_csv",
        required=True,
        help="Path to 500ppm CSV (expects a compound column, e.g. 'Compound No').",
    )
    parser.add_argument(
        "--class_compound_col",
        default="compound",
        help="Compound ID column in class metadata. Default: 'compound'.",
    )
    parser.add_argument(
        "--label_col",
        default="chemical_class",
        help="Class label column in class metadata. Default: 'chemical_class'.",
    )
    parser.add_argument(
        "--min_compounds_per_class",
        type=int,
        default=30,
        help="Minimum compounds per class for selection. Default: 30.",
    )
    parser.add_argument(
        "--ppm500_compound_col",
        default="Compound No",
        help="Compound ID column name in 500ppm CSV. Default: 'Compound No'.",
    )
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def load_compound_image_counts(metadata_path: Path) -> dict[str, tuple[int, int, int]]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    counts: dict[str, tuple[int, int, int]] = {}

    for entry in metadata:
        compound_id = str(entry.get("Compound", "")).strip()
        if not compound_id:
            continue

        treated_total = 0
        control_total = 0
        for key, plate in entry.items():
            if key == "Compound":
                continue
            treated = plate.get("treated", []) or []
            control = plate.get("control", []) or []
            treated_total += len(treated)
            control_total += len(control)

        counts[compound_id] = (treated_total, control_total, treated_total + control_total)

    return counts


def select_class_filtered_compounds(
    class_metadata_path: Path,
    compound_col: str,
    label_col: str,
    min_compounds_per_class: int,
) -> tuple[set[str], int, int]:
    df = _read_table(class_metadata_path)

    missing = [c for c in [compound_col, label_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns in class metadata: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    sdf = df[[compound_col, label_col]].copy()
    sdf[compound_col] = sdf[compound_col].astype(str).str.strip()
    sdf[label_col] = sdf[label_col].astype(str).str.strip()
    sdf = sdf[(sdf[compound_col] != "") & (sdf[label_col] != "")]

    # Keep one row per compound for class counting.
    sdf_unique = sdf.drop_duplicates(subset=[compound_col], keep="first")

    class_counts = sdf_unique[label_col].value_counts()
    selected_classes = set(class_counts[class_counts >= min_compounds_per_class].index.tolist())

    selected_compounds = set(
        sdf_unique.loc[sdf_unique[label_col].isin(selected_classes), compound_col].tolist()
    )

    return selected_compounds, len(selected_classes), len(sdf_unique)


def compounds_from_500ppm(csv_path: Path, compound_col: str) -> tuple[set[str], int]:
    df = pd.read_csv(csv_path)
    if compound_col not in df.columns:
        raise ValueError(
            f"Column '{compound_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    compounds = set(df[compound_col].astype(str).str.strip().tolist())
    compounds.discard("")
    return compounds, len(df)


def summarize_image_counts(
    compounds: set[str],
    image_counts: dict[str, tuple[int, int, int]],
) -> tuple[int, int, int, int, int]:
    treated = 0
    control = 0
    found = 0

    for cid in compounds:
        if cid not in image_counts:
            continue
        t, c, _ = image_counts[cid]
        treated += t
        control += c
        found += 1

    return treated + control, treated, control, found, len(compounds) - found


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.chemical_metadata)
    class_metadata_path = Path(args.class_metadata)
    csv_path = Path(args.ppm500_csv)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Chemical metadata JSON not found: {metadata_path}")
    if not class_metadata_path.exists():
        raise FileNotFoundError(f"Class metadata file not found: {class_metadata_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"500ppm CSV not found: {csv_path}")

    image_counts = load_compound_image_counts(metadata_path)
    selected_compounds, _, _ = select_class_filtered_compounds(
        class_metadata_path=class_metadata_path,
        compound_col=args.class_compound_col,
        label_col=args.label_col,
        min_compounds_per_class=args.min_compounds_per_class,
    )
    ppm500_compounds, _ = compounds_from_500ppm(
        csv_path,
        args.ppm500_compound_col,
    )

    sel_total, sel_treated, sel_control, _, _ = summarize_image_counts(
        selected_compounds,
        image_counts,
    )
    ppm_total, ppm_treated, ppm_control, _, _ = summarize_image_counts(
        ppm500_compounds,
        image_counts,
    )

    print("=" * 60)
    print("Chemical-class selected compounds (class filter)")
    print("=" * 60)
    print(f"Treated images       : {sel_treated}")
    print(f"Control images       : {sel_control}")
    print(f"Total images         : {sel_total}")

    print("\n" + "=" * 60)
    print("500ppm CSV compounds")
    print("=" * 60)
    print(f"Treated images       : {ppm_treated}")
    print(f"Control images       : {ppm_control}")
    print(f"Total images         : {ppm_total}")


if __name__ == "__main__":
    main()
