import re
from pathlib import Path

import pandas as pd

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# Order matters: earlier splits take priority when removing cross-split duplicates
SPLITS = ["train", "dev", "test"]

EC_PATTERN = re.compile(r"EC:(\d+)")


def extract_main_ec(value):
    """Returns the first-level EC class as 'EC:<n>', or None if no EC number is found."""
    match = EC_PATTERN.search(str(value))
    if match:
        return f"EC:{match.group(1)}"
    return None


def clean_split(df):
    """Adds main_ec, drops missing/empty rows, and removes duplicate sequences."""
    df = df.copy()
    df["main_ec"] = df["labels_str"].apply(extract_main_ec)

    # Missing or empty sequences / labels
    df["seq"] = df["seq"].astype("string").str.strip()
    df = df.dropna(subset=["seq", "main_ec"])
    df = df[df["seq"].str.len() > 0]

    # Sequences that appear with conflicting labels are ambiguous, so drop them entirely
    n_labels = df.groupby("seq")["main_ec"].transform("nunique")
    df = df[n_labels == 1]

    # Identical sequence + identical label: keep one copy
    df = df.drop_duplicates(subset="seq", keep="first")

    return df.reset_index(drop=True)


def process():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    seen_sequences = set()  # sequences already assigned to an earlier split

    for split in SPLITS:
        input_file = RAW_DIR / f"SwissProt-EC-{split}.csv"
        output_file = PROCESSED_DIR / f"SwissProt-EC-{split}_updated.csv"

        df = pd.read_csv(input_file)
        input_len = len(df)

        df = clean_split(df)
        after_clean_len = len(df)

        # Remove sequences that already appear in an earlier split (prevents leakage)
        df = df[~df["seq"].isin(seen_sequences)].reset_index(drop=True)
        seen_sequences.update(df["seq"])

        df.to_csv(output_file, index=False)

        print(
            f"{split}: {input_len} rows -> {after_clean_len} after cleaning/dedup "
            f"-> {len(df)} after cross-split removal, saved to {output_file}"
        )
        print(df["main_ec"].value_counts().sort_index().to_string())
        print()


if __name__ == "__main__":
    process()
