import csv
from glob import glob
import os
from pathlib import Path

from madress_2023 import constants
from madress_2023.constants import STANDARDIZED_CSV_HEADER, CsvInfo
from madress_2023.utils.full_path import full_path
from madress_2023.utils.normalization import (
    norm_ad_transform,
    norm_education_transform,
    norm_gender_transform,
    NORM_MMSE_TRANSFORM,
)


def transform_csv(in_path: Path, out_feat_dir, csv_info: CsvInfo):

    # Make sure input path exists.
    in_dir = in_path.parent
    if not in_path.exists():
        raise Exception("Path does not exist: %s" % str(in_path))

    # Calculate relative path from DIR_PROJECT to in_dir.
    in_dir_rel_path = os.path.relpath(in_dir, constants.DIR_PROJECT)
    out_feat_dir_rel_path = os.path.relpath(out_feat_dir, constants.DIR_PROJECT)

    # Calculate relative path from in_dir to audio root_path.
    audio_root_rel_path = os.path.relpath(str(csv_info.audio_root_path), in_dir)

    # Open raw CSV file.
    with open(in_path, encoding="utf8", mode="r") as in_csv:

        # Create CSV reader/writer.
        csv_reader = csv.reader(in_csv)

        # Initialize resulting rows.
        out_rows = []

        # Iterate through raw CSV...
        for idx, in_row in enumerate(csv_reader):

            # Write header row.
            if idx == 0:
                out_rows.append(STANDARDIZED_CSV_HEADER)
                continue

            # Skip empty row.
            if len(in_row) == 0:
                continue

            # Process row...
            out_row = []

            # 0. audio path
            #    Note: audio files will remain in the "raw" directory (in_dir).
            audio_fname = in_row[csv_info.col_audio_fname]
            audio_rel_path = os.path.join(audio_root_rel_path, audio_fname + ".*")
            audio_base, _ = os.path.splitext(str(audio_rel_path))
            path = os.path.join(in_dir_rel_path, str(audio_rel_path))
            path = os.path.relpath(path)  # resolve path

            # determine extension
            _full_path = full_path(path)
            matches = glob(_full_path)
            if len(matches) == 0:
                raise Exception(
                    f"Could not find the audio file: {path}. \nFull path: {_full_path}"
                )
            ext = os.path.splitext(matches[0])[1]
            path = path.replace(".*", ext)
            out_row.append(path)

            # 1. egemaps path
            egemaps_path = audio_base + ".egemaps.pt"
            egemaps_path = os.path.join(out_feat_dir_rel_path, egemaps_path)
            egemaps_path = os.path.relpath(egemaps_path)  # resolve path
            out_row.append(egemaps_path)

            # 2-4. age, gender, education
            age = in_row[csv_info.col_age]
            gender = norm_gender_transform(in_row[csv_info.col_gender])
            education = norm_education_transform(in_row[csv_info.col_education])
            out_row.append(age)
            out_row.append(str(gender))
            out_row.append(str(education))

            # 5-7. AD, MMSE, norm_MMSE
            if csv_info.col_ad is not None:
                ad = norm_ad_transform(in_row[csv_info.col_ad])
            else:
                ad = -1 # Test CSV has no AD information
            if csv_info.col_mmse is not None:
                mmse = in_row[csv_info.col_mmse]
                norm_mmse = NORM_MMSE_TRANSFORM.transform_str(mmse)
            else:
                mmse = -1 # Test CSV has no MMSE information
                norm_mmse = -1
            out_row.append(str(ad))
            out_row.append(str(mmse))
            out_row.append(str(norm_mmse))

            # Append to output rows.
            out_rows.append(out_row)

    return out_rows
