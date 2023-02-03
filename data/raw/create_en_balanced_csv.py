import csv
import os
from random import Random

script_dir = os.path.dirname(__file__)
en_full_csv_path = os.path.join(script_dir, "training-groundtruth.csv")
out_csv_path = os.path.join(script_dir, "en-balanced-groundtruth.csv")
train_out_csv_path = os.path.join(script_dir, "en-balanced-train-groundtruth.csv")
val_out_csv_path = os.path.join(script_dir, "en-balanced-val-groundtruth.csv")

rows_ad = []
rows_control = []
with open(en_full_csv_path, mode="r") as f:
    csv_reader = csv.reader(f)
    for idx, row in enumerate(csv_reader):
        if idx == 0 or len(row) == 0:
            continue  # skip header and blank rows

        fname, age, gender, edu, diag, mmse = row
        if edu == "NA":
            edu = 12  # keep rows with missing education, replace with 12 years
        if mmse == "NA":
            continue  # exclude adrso256 who has no MMSE

        # Convert numeric (for printing later)
        _row = [fname, int(age), gender, int(edu), diag, int(mmse)]

        if diag == "ProbableAD":
            rows_ad.append(_row)
        else:
            rows_control.append(_row)

# num AD > num control ==> include all controls and subset of AD cases.
rows_ad.sort(key=lambda x: (int(x[-1]), int(x[1])))  # sort by (MMSE, age)

# Include first X, last X, then every other
ad_subset = []
for i in range(len(rows_ad)):
    if i < 49:
        ad_subset.append(rows_ad[i])
    elif i >= len(rows_control) - 50:
        ad_subset.append(rows_ad[i])
    else:
        if i % 2 == 0:
            ad_subset.append(rows_ad[i])
rows_ad = ad_subset

# Check if inclusion code produces an equal number of controls and AD.
if len(rows_control) != len(rows_ad):
    _err_msg = (
        f"Adjust balancing algorithm: #AD={len(rows_ad)}, #control={len(rows_control)}"
    )
    raise Exception(_err_msg)

balanced_rows = rows_ad + rows_control
balanced_rows.sort(key=lambda x: x[0])  # filename

# Write to output CSV.
_header = ["adressfname", "age", "gender", "educ", "dx", "mmse"]
balanced_rows.insert(0, _header)
with open(out_csv_path, mode="w") as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in balanced_rows:
        csv_writer.writerow(row)


# Also create train/val partitions within the balanced set.
rdm = Random(42)  # Reproducible random number generation.
rdm.shuffle(rows_ad)
rdm.shuffle(rows_control)
frac_train = 0.80
n_train = int(len(rows_ad) * frac_train)
balanced_train_rows = rows_ad[:n_train] + rows_control[:n_train]
balanced_val_rows = rows_ad[n_train:] + rows_control[n_train:]
balanced_train_rows.sort(key=lambda x: x[0])  # filename
balanced_val_rows.sort(key=lambda x: x[0])  # filename
balanced_train_rows.insert(0, _header)
balanced_val_rows.insert(0, _header)

# Write to output CSV.
with open(train_out_csv_path, mode="w") as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in balanced_train_rows:
        csv_writer.writerow(row)
# Write to output CSV.
with open(val_out_csv_path, mode="w") as f:
    csv_writer = csv.writer(f, quoting=csv.QUOTE_NONNUMERIC)
    for row in balanced_val_rows:
        csv_writer.writerow(row)
