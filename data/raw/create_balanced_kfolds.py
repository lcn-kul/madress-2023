import csv
import os
from random import Random

script_dir = os.path.dirname(__file__)
gr_sample_csv_path = os.path.join(script_dir, "sample-gr-groundtruth.csv")
gr_out_csv_path_pattern = os.path.join(
    script_dir, "sample-gr-%s-%ifolds-%i-groundtruth.csv"
)


def create_folds(rows, num_folds):
    sections = []
    for i in range(num_folds):
        _section = rows[i::num_folds]
        sections.append(_section)
    train_sets = []
    val_sets = []
    for i in range(num_folds):
        train_sets.append([])
        val_sets.append([])
        for j in range(num_folds):
            if i == j:
                val_sets[-1].extend(sections[j])
            else:
                train_sets[-1].extend(sections[j])
    return train_sets, val_sets


csv_path = gr_sample_csv_path
out_pattern = gr_out_csv_path_pattern

# Load CSV, separate AD and control.
rows_ad = []
rows_control = []
_header = None
with open(csv_path, mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0:
            _header = row
            continue  # skip header row
        if len(row) == 0:
            continue
        fname, age, gender, edu, diag, mmse = row

        # Convert numeric (for printing later)
        _row = [fname, int(age), gender, int(edu), diag, int(mmse)]

        if diag == "ProbableAD":
            rows_ad.append(_row)
        else:
            rows_control.append(_row)

# Sort both by MMSE/age for some uniformity in folds.
rows_ad.sort(key=lambda x: (int(x[-1]), int(x[1])))
rows_control.sort(key=lambda x: (int(x[-1]), int(x[1])))


# Create 2-folds.
num_folds = 2
ad_2folds_trains, ad_2folds_vals = create_folds(rows_ad, num_folds)
control_2folds_trains, control_2folds_vals = create_folds(rows_control, num_folds)

# Write to CSV.
for i in range(2):
    fold_train_path = out_pattern % ("train", 2, i)
    rows_train = ad_2folds_trains[i] + control_2folds_trains[i]
    rows_train.insert(0, _header)
    fold_val_path = out_pattern % ("val", 2, i)
    rows_val = ad_2folds_vals[i] + control_2folds_vals[i]
    rows_val.insert(0, _header)
    with open(fold_train_path, mode="w") as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in rows_train:
            csv_writer.writerow(row)
    with open(fold_val_path, mode="w") as f:
        csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
        for row in rows_val:
            csv_writer.writerow(row)
