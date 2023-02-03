from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List

from madress_2023.utils.normalization import NormTransform, NORM_MMSE_TRANSFORM


# =============================== #
#              PATHS              #
# =============================== #

# Locate the root directory of the project.
# madress-2023/
DIR_PROJECT = None
for path in Path(__file__).parents:
    if path.name == "madress-2023":
        DIR_PROJECT = path
        break
if DIR_PROJECT is None:
    raise Exception("Unable to locate root dir.")

# madress-2023/data/
DIR_DATA = DIR_PROJECT.joinpath("data")
DIR_DATA_PROCESSED = DIR_DATA.joinpath("processed")
DIR_DATA_RAW = DIR_DATA.joinpath("raw")

# madress-2023/logs/
DIR_LOGS = DIR_PROJECT.joinpath("logs")
DIR_LOGS.mkdir(parents=True, exist_ok=True)

# madress-2023/models/
DIR_MODELS = DIR_PROJECT.joinpath("models")

# =============================== #
#              SPLITS             #
# =============================== #


class Split(Enum):
    EN_BALANCED_TRAIN = 0
    EN_BALANCED_VAL = 1
    GR_SAMPLE = 2
    GR_SAMPLE_2FOLDS_TRAIN = 3
    GR_SAMPLE_2FOLDS_VAL = 4
    GR_TEST = 5


ALL_SPLITS = [
    Split.EN_BALANCED_TRAIN,
    Split.EN_BALANCED_VAL,
    Split.GR_SAMPLE,
    Split.GR_SAMPLE_2FOLDS_TRAIN,
    Split.GR_SAMPLE_2FOLDS_VAL,
    Split.GR_TEST,
]

# =============================== #
#            RAW CSVS             #
# =============================== #


# Information on how to parse raw CSVs.
class CsvInfo:
    """Information about the ground-truth CSV file. Note that the columns are
    zero-indexed.
    """

    def __init__(
        self,
        csv_path: Path,
        col_audio_fname: int,
        col_age: int,
        col_gender: int,
        col_education: int,
        col_ad: int,
        col_mmse: int,
        audio_root_path: Path = None,
    ):
        assert csv_path is not None
        self.csv_path = csv_path
        self.col_audio_fname = col_audio_fname
        self.col_age = col_age
        self.col_gender = col_gender
        self.col_education = col_education
        self.col_ad = col_ad
        self.col_mmse = col_mmse
        if audio_root_path is None:
            self.audio_root_path = csv_path.parent
        else:
            self.audio_root_path = audio_root_path


# Columns are the same for all raw CSVs.
_csv_cols = {
    "col_audio_fname": 0,
    "col_age": 1,
    "col_gender": 2,
    "col_education": 3,
    "col_ad": 4,
    "col_mmse": 5,
}

# Audio root directories.
en_audio_root = DIR_DATA_RAW.joinpath("train")
gr_sample_audio_root = DIR_DATA_RAW.joinpath("sample-gr")
gr_test_audio_root = DIR_DATA_RAW.joinpath("test-gr")

# RAW CSV INFO: en_balanced_train
RAW_EN_BALANCED_TRAIN_CSV_INFO = CsvInfo(
    csv_path=DIR_DATA_RAW.joinpath("en-balanced-train-groundtruth.csv"),
    audio_root_path=en_audio_root,
    **_csv_cols,
)

# RAW CSV INFO: en_balanced_val
RAW_EN_BALANCED_VAL_CSV_INFO = CsvInfo(
    csv_path=DIR_DATA_RAW.joinpath("en-balanced-val-groundtruth.csv"),
    audio_root_path=en_audio_root,
    **_csv_cols,
)

# RAW CSV INFO: gr_sample
RAW_GR_SAMPLE_CSV_INFO = CsvInfo(
    csv_path=DIR_DATA_RAW.joinpath("sample-gr-groundtruth.csv"),
    audio_root_path=gr_sample_audio_root,
    **_csv_cols,
)

# RAW CSV INFO: gr_sample_2folds
RAW_GR_SAMPLE_2FOLDS_TRAIN_CSV_INFO: List[CsvInfo] = []
RAW_GR_SAMPLE_2FOLDS_VAL_CSV_INFO: List[CsvInfo] = []
for fold in range(2):
    x = CsvInfo(
        csv_path=DIR_DATA_RAW.joinpath(
            f"sample-gr-train-2folds-{fold}-groundtruth.csv"
        ),
        audio_root_path=gr_sample_audio_root,
        **_csv_cols,
    )
    RAW_GR_SAMPLE_2FOLDS_TRAIN_CSV_INFO.append(x)
    x = CsvInfo(
        csv_path=DIR_DATA_RAW.joinpath(f"sample-gr-val-2folds-{fold}-groundtruth.csv"),
        audio_root_path=gr_sample_audio_root,
        **_csv_cols,
    )
    RAW_GR_SAMPLE_2FOLDS_VAL_CSV_INFO.append(x)

# RAW CSV INFO: gr_test
gr_test_audio_root = DIR_DATA_RAW.joinpath("test-gr")
RAW_GR_TEST_CSV_INFO = CsvInfo(
    csv_path=DIR_DATA_RAW.joinpath("ADReSS-M-meta.csv"),
    audio_root_path=gr_test_audio_root,
    col_audio_fname=0,
    col_age=1,
    col_gender=2,
    col_education=3,
    col_ad=None,
    col_mmse=None,
)



# ============== #
# PROCESSED DIRS #
# ============== #

# STANDARDIZED CSV FORMAT
# ==> each raw CSV will be transformed into this format and will go in the folder
#     `data/processed/`
@dataclass
class StandardizedCsvInfo:
    """Information about the standardized CSV file. Note that the columns are
    zero-indexed.
    """

    col_audio_path: int = 0  # audio path
    col_egemaps_path: int = 1
    col_age: int = 2
    col_gender: int = 3
    col_education: int = 4
    col_ad: int = 5
    col_mmse: int = 6
    col_norm_mmse: int = 7


STANDARDIZED_CSV_INFO = StandardizedCsvInfo()
STANDARDIZED_CSV_HEADER = [
    "audio_path",
    "egemaps_path",
    "age",
    "gender",
    "education",
    "ad",
    "mmse",
    "norm_mmse",
]


class DatasetDir:
    """Structure of each dataset split directory."""

    root_dir: Path

    def __init__(self, root_dir: Path, csv_name: str = None) -> None:
        self.root_dir = root_dir
        if csv_name is None:
            self.csv_path = root_dir.joinpath("data.csv")
        else:
            self.csv_path = root_dir.joinpath(csv_name)
        self.features_dir = root_dir.joinpath("features")
        self.create_dirs()

    def create_dirs(self):
        self.features_dir.mkdir(mode=0o755, parents=True, exist_ok=True)


# Full datasets.
DATASET_EN_BALANCED_TRAIN = DatasetDir(
    DIR_DATA_PROCESSED.joinpath("en_balanced"), "data_train.csv"
)
DATASET_EN_BALANCED_VAL = DatasetDir(
    DIR_DATA_PROCESSED.joinpath("en_balanced"), "data_val.csv"
)
DATASET_GR_SAMPLE = DatasetDir(DIR_DATA_PROCESSED.joinpath("gr_sample"))
DATASET_GR_SAMPLE_2FOLDS_TRAIN: List[DatasetDir] = []
DATASET_GR_SAMPLE_2FOLDS_VAL: List[DatasetDir] = []
for fold in range(2):
    x = DatasetDir(
        DIR_DATA_PROCESSED.joinpath("gr_sample"), f"data_train_2folds_{fold}.csv"
    )
    DATASET_GR_SAMPLE_2FOLDS_TRAIN.append(x)
    x = DatasetDir(
        DIR_DATA_PROCESSED.joinpath("gr_sample"), f"data_val_2folds_{fold}.csv"
    )
    DATASET_GR_SAMPLE_2FOLDS_VAL.append(x)
DATASET_GR_TEST = DatasetDir(DIR_DATA_PROCESSED.joinpath("gr_test"))


def get_dataset(split: Split, fold: int = None):
    if split == Split.EN_BALANCED_TRAIN:
        return DATASET_EN_BALANCED_TRAIN
    if split == Split.EN_BALANCED_VAL:
        return DATASET_EN_BALANCED_VAL
    if split == Split.GR_SAMPLE:
        return DATASET_GR_SAMPLE
    if split == Split.GR_SAMPLE_2FOLDS_TRAIN:
        assert fold is not None
        return DATASET_GR_SAMPLE_2FOLDS_TRAIN[fold]
    if split == Split.GR_SAMPLE_2FOLDS_VAL:
        assert fold is not None
        return DATASET_GR_SAMPLE_2FOLDS_VAL[fold]
    if split == Split.GR_TEST:
        return DATASET_GR_TEST
    raise Exception(f"Unknown split: {split.name}")


def get_csv_info(split: Split, fold: int = None):
    if split == Split.EN_BALANCED_TRAIN:
        return RAW_EN_BALANCED_TRAIN_CSV_INFO
    if split == Split.EN_BALANCED_VAL:
        return RAW_EN_BALANCED_VAL_CSV_INFO
    if split == Split.GR_SAMPLE:
        return RAW_GR_SAMPLE_CSV_INFO
    if split == Split.GR_SAMPLE_2FOLDS_TRAIN:
        assert fold is not None
        return RAW_GR_SAMPLE_2FOLDS_TRAIN_CSV_INFO[fold]
    if split == Split.GR_SAMPLE_2FOLDS_VAL:
        assert fold is not None
        return RAW_GR_SAMPLE_2FOLDS_VAL_CSV_INFO[fold]
    if split == Split.GR_TEST:
        return RAW_GR_TEST_CSV_INFO
    raise Exception(f"Unknown split: {split.name}")
