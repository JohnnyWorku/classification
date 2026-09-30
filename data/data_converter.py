from pathlib import Path

from datasets import load_dataset

DATASET_NAME = "DanielHesslow/SwissProt-EC"
OUTPUT_DIR = Path("data/raw")


def convert_splits_to_csv(output_dir=OUTPUT_DIR, overwrite=False):
    """Downloads the SwissProt-EC dataset and saves each split as a CSV file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Skip the download entirely if every split already exists
    expected = {name: output_dir / f"SwissProt-EC-{name}.csv" for name in ("train", "validation", "test")}
    if not overwrite and all(path.exists() for path in expected.values()):
        print("All CSV files already exist, skipping download.")
        return

    dataset = load_dataset(DATASET_NAME)

    for split_name, split_data in dataset.items():
        csv_path = output_dir / f"SwissProt-EC-{split_name}.csv"
        split_data.to_csv(csv_path, index=False)
        print(f"{split_name} ({len(split_data)} rows) saved to {csv_path}")


if __name__ == "__main__":
    convert_splits_to_csv()