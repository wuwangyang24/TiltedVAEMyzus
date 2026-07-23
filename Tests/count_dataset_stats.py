"""
Simple dataset counter for:
1) Chemical-class prediction metadata JSON
2) 500ppm efficacy CSV

Examples:
python Tests/count_dataset_stats.py \
  --chemical_metadata METADATA/metadata_compound_all100ppm.json \
  --ppm500_csv Tests/efficacy500_classifier/compounds500ppm.csv
"""

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Count total images in chemical-class metadata and compounds in 500ppm CSV."
    )
    parser.add_argument(
        "--chemical_metadata",
        required=True,
        help="Path to chemical-class metadata JSON (compound -> plate -> treated/control paths).",
    )
    parser.add_argument(
        "--ppm500_csv",
        required=True,
        help="Path to 500ppm CSV (expects a compound column, e.g. 'Compound No').",
    )
    parser.add_argument(
        "--compound_col",
        default="Compound No",
        help="Compound ID column name in 500ppm CSV. Default: 'Compound No'.",
    )
    return parser.parse_args()


def count_images_in_metadata(metadata_path: Path) -> tuple[int, int, int, int]:
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    n_compounds = 0
    n_plates = 0
    n_treated_images = 0
    n_control_images = 0

    for entry in metadata:
        n_compounds += 1
        for key, plate in entry.items():
            if key == "Compound":
                continue
            n_plates += 1
            treated = plate.get("treated", []) or []
            control = plate.get("control", []) or []
            n_treated_images += len(treated)
            n_control_images += len(control)

    total_images = n_treated_images + n_control_images
    return total_images, n_treated_images, n_control_images, n_compounds


def count_compounds_in_500ppm(csv_path: Path, compound_col: str) -> tuple[int, int]:
    df = pd.read_csv(csv_path)
    if compound_col not in df.columns:
        raise ValueError(
            f"Column '{compound_col}' not found in {csv_path}. "
            f"Available columns: {list(df.columns)}"
        )

    compounds = df[compound_col].astype(str).str.strip()
    n_rows = len(df)
    n_unique_compounds = compounds.nunique(dropna=True)
    return n_rows, n_unique_compounds


def main() -> None:
    args = parse_args()

    metadata_path = Path(args.chemical_metadata)
    csv_path = Path(args.ppm500_csv)

    if not metadata_path.exists():
        raise FileNotFoundError(f"Chemical metadata JSON not found: {metadata_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"500ppm CSV not found: {csv_path}")

    total_images, treated_images, control_images, n_compounds = count_images_in_metadata(metadata_path)
    n_rows, n_unique_compounds = count_compounds_in_500ppm(csv_path, args.compound_col)

    print("=" * 60)
    print("Chemical-class dataset")
    print("=" * 60)
    print(f"Metadata file        : {metadata_path}")
    print(f"Compounds            : {n_compounds}")
    print(f"Treated images       : {treated_images}")
    print(f"Control images       : {control_images}")
    print(f"Total images         : {total_images}")

    print("\n" + "=" * 60)
    print("500ppm CSV")
    print("=" * 60)
    print(f"CSV file             : {csv_path}")
    print(f"Rows                 : {n_rows}")
    print(f"Unique compounds     : {n_unique_compounds}")


if __name__ == "__main__":
    main()
