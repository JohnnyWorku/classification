from datasets import load_dataset


dataset = load_dataset("DanielHesslow/SwissProt-EC")

for split_name, split_data in dataset.items():
    csv_filename = f"SwissProt-EC-{split_name}.csv"

    split_data.to_csv(csv_filename, index=False)
    print(f"{split_name} converted to {csv_filename} successfully.")
