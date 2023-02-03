import csv
import librosa
import numpy as np
from opensmile.core.smile import Smile
from opensmile.core.define import FeatureSet, FeatureLevel
import os
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm
import warnings

from madress_2023 import constants
from madress_2023.constants import Split, STANDARDIZED_CSV_INFO
from madress_2023.train.config import FEAT_SEQ_LEN
from madress_2023.utils.full_path import full_path


def _decode_non_mp3_file_like(file, new_sr):
    # Source:
    # https://huggingface.co/docs/datasets/_modules/datasets/features/audio.html#Audio
    # array, sampling_rate = sf.read(file)
    array, sampling_rate = librosa.load(file, sr=new_sr) # works with MP3 if ffmpeg is installed
    array = array.T
    array = librosa.to_mono(array)
    if new_sr and new_sr != sampling_rate:
        array = librosa.resample(
            array, orig_sr=sampling_rate, target_sr=new_sr, res_type="kaiser_best"
        )
        sampling_rate = new_sr
    return array, sampling_rate


def load_audio(file_path: str, sampling_rate: int) -> torch.Tensor:
    array, _ = _decode_non_mp3_file_like(file_path, sampling_rate)
    array = np.float32(array)
    return array


class SimpleCsvDataset(Dataset):
    def __init__(self, split: Split, fold: int = None):
        super().__init__()

        # Returns a constants.DatasetDir containing information about the dataset.
        dataset = constants.get_dataset(split, fold)

        # Load CSV.
        self.csv_data = []  # feature_path, norm_mos
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):
                if idx == 0:
                    continue  # Skip header row.

                # Save feature_path, norm_mos
                audio_path = in_row[STANDARDIZED_CSV_INFO.col_audio_path]
                egemaps_path = in_row[STANDARDIZED_CSV_INFO.col_egemaps_path]
                self.csv_data.append([audio_path, egemaps_path])

    def __len__(self):
        return len(self.csv_data)

    def __getitem__(self, index):
        audio_path, egemaps_path = self.csv_data[index]
        return audio_path, egemaps_path


def extract_features(
    split: Split,
    fold: int = None,
):

    # For printing...
    split_name = str(split).lower().split(".")[1]
    print(f"Extracting features for {split_name} split.")

    # Create dataset.
    csv_dataset = SimpleCsvDataset(split, fold)
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=None,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
    )

    print(f"Calculating features for {len(csv_dataset)} audio files...")
    smile_lld = Smile(
        feature_set=FeatureSet.eGeMAPSv02,
        feature_level=FeatureLevel.LowLevelDescriptors,
    )
    warnings.simplefilter('ignore')
    for audio_path, egemaps_path in tqdm(csv_dataloader):

        # Calculate egemaps over the N segments (N=FEAT_SEQ_LEN)
        if os.path.exists(full_path(egemaps_path)):
            continue
        else:
            sr = 16000
            audio_np = load_audio(full_path(audio_path), sampling_rate=sr)
            x = (audio_np.shape[0] // FEAT_SEQ_LEN) * FEAT_SEQ_LEN
            audio_segments = np.split(audio_np[:x], FEAT_SEQ_LEN)
            _egemaps = []
            for audio_segment in audio_segments:
                _, _, y = smile_lld.process(audio_segment, sr)
                _egemaps.append(torch.from_numpy(y[0, :]))
            egemaps = torch.stack(_egemaps, dim=0)
            torch.save(egemaps, full_path(egemaps_path))
    warnings.simplefilter('always')

    print("")
    print(f"Finished.")
