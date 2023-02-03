import csv

from madress_2023 import constants
from madress_2023.constants import ALL_SPLITS, STANDARDIZED_CSV_HEADER, Split
from madress_2023.csv.transform_csv import transform_csv


def process_raw_csv(split: Split, fold: int = None):

    if fold is None and "2FOLDS" in split.name:
        for _fold in range(2):
            process_raw_csv(split, _fold)
        return

    # Returns a constants.DatasetDir containing information about the dataset.
    dataset = constants.get_dataset(split, fold)
    csv_info = constants.get_csv_info(split, fold)

    # Print split name.
    split_name = str(split).lower().split(".")[1]
    print(f"Processing raw CSVs for split: {split_name}.")

    # Load all CSVs.
    rows = []
    csv_path = csv_info.csv_path

    print(f"Processing raw CSV: {csv_path}")
    rows = transform_csv(
        in_path=csv_path,
        out_feat_dir=dataset.features_dir,
        csv_info=csv_info,
    )

    # Replace header row.
    rows.pop(0)
    rows.insert(0, STANDARDIZED_CSV_HEADER)

    # Write to output CSV.
    with open(dataset.csv_path, mode="w", encoding="utf-8") as f_out:
        csv_writer = csv.writer(f_out)
        csv_writer.writerows(rows)

    print("Finished.")


if __name__ == "__main__":
    for split in ALL_SPLITS:
        process_raw_csv(split)
