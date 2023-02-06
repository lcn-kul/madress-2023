import csv
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from typing import List, Tuple

from madress_2023 import constants
from madress_2023.constants import Split, STANDARDIZED_CSV_INFO
from madress_2023.train.model_pl import ModelPL
from madress_2023.utils.full_path import full_path


class CsvDataset(Dataset):

    def __init__(
        self,
        split: Split,
        times_per_epoch: int = 1,
        do_rdm_greek: bool = False,
        fold = None,
        only_greek: bool = False,
        ad_models: List[ModelPL] = None,
        device="cuda",
    ) -> None:
        super().__init__()

        self.split = split
        assert times_per_epoch > 0
        self.times_per_epoch = times_per_epoch
        self.device = device
        self.only_greek = only_greek
        if do_rdm_greek:
            assert fold is not None
            self.fold = fold
        if only_greek:
            assert do_rdm_greek
        self.do_rdm_greek = do_rdm_greek

        self.ad_models = ad_models
        if ad_models is not None:
            for x in ad_models:
                x.eval()


        # For printing...
        split_name = str(split).lower().split(".")[1]
        print(f"Creating dataloader for {split_name} set.")

        dataset = constants.get_dataset(split, fold)
        # Load CSV.
        self.csv_data = []
        self.ram_features = []
        with open(dataset.csv_path, encoding="utf8", mode="r") as in_csv:
            csv_reader = csv.reader(in_csv)
            for idx, in_row in enumerate(csv_reader):

                # Skip header row.
                if idx == 0:
                    continue

                # save age, gender
                col_age = STANDARDIZED_CSV_INFO.col_age
                col_gender = STANDARDIZED_CSV_INFO.col_gender
                col_educ = STANDARDIZED_CSV_INFO.col_education
                age = float(in_row[col_age])
                gender = float(in_row[col_gender])
                educ = float(in_row[col_educ])

                # Save ad, norm_mmse
                col_ad = STANDARDIZED_CSV_INFO.col_ad
                col_n_mmse = STANDARDIZED_CSV_INFO.col_norm_mmse
                ad = torch.tensor(int(in_row[col_ad]), dtype=torch.int64)
                norm_mmse = torch.tensor(float(in_row[col_n_mmse]))

                self.csv_data.append([age, gender, educ, ad, norm_mmse])

                # Load egemaps features.
                col_egemaps_path = STANDARDIZED_CSV_INFO.col_egemaps_path
                egemaps_path = in_row[col_egemaps_path]
                egemaps = torch.load(full_path(egemaps_path))
                self.ram_features.append(egemaps)
        self.num_ram_features = len(self.ram_features)

        # Load gr CSV.
        if self.do_rdm_greek:
            if split == Split.EN_BALANCED_TRAIN:
                split_rdm_greek = Split.GR_SAMPLE_2FOLDS_TRAIN
            elif split == Split.EN_BALANCED_VAL:
                split_rdm_greek = Split.GR_SAMPLE_2FOLDS_VAL
            else:
                raise Exception("something wrong with splits")
            dataset_gr = constants.get_dataset(split_rdm_greek, fold)
            self.csv_data_gr = []
            self.ram_features_gr = []
            with open(dataset_gr.csv_path, encoding="utf8", mode="r") as in_csv:
                csv_reader = csv.reader(in_csv)
                for idx, in_row in enumerate(csv_reader):

                    # Skip header row.
                    if idx == 0:
                        continue

                    # save age, gender, educ
                    col_age = STANDARDIZED_CSV_INFO.col_age
                    col_gender = STANDARDIZED_CSV_INFO.col_gender
                    col_educ = STANDARDIZED_CSV_INFO.col_education
                    age = float(in_row[col_age])
                    gender = float(in_row[col_gender])
                    educ = float(in_row[col_educ])

                    # Save ad, norm_mmse
                    col_ad = STANDARDIZED_CSV_INFO.col_ad
                    col_n_mmse = STANDARDIZED_CSV_INFO.col_norm_mmse
                    ad = torch.tensor(int(in_row[col_ad]), dtype=torch.int64)
                    norm_mmse = torch.tensor(float(in_row[col_n_mmse]))

                    self.csv_data_gr.append([age, gender, educ, ad, norm_mmse])

                    # Load egemaps features.
                    col_egemaps_path = STANDARDIZED_CSV_INFO.col_egemaps_path
                    egemaps_path = in_row[col_egemaps_path]
                    egemaps = torch.load(full_path(egemaps_path))
                    self.ram_features_gr.append(egemaps)

        # Cache for AD predictions.
        self.ad_cache = [None] * len(self.csv_data)
        if self.do_rdm_greek:
            self.gr_ad_cache = [None] * len(self.csv_data_gr)

        print("Finished.")

    def __len__(self):
        if self.only_greek:
            return len(self.csv_data_gr) * self.times_per_epoch 
        else:
            return len(self.csv_data) * self.times_per_epoch

    def _get_ad_prob(self, features):
        assert self.ad_models is not None
        N = len(self.ad_models)
        ad_probs = torch.zeros((N,))
        for idx in range(N):
            with torch.no_grad():
                model = self.ad_models[idx].model
                _feat_ad = tuple(f.unsqueeze(0) for f in features)
                _f, _ = self.ad_models[idx]._preprocess_batch((_feat_ad, None))
                out_ad = model.forward(_f)
                probs = F.softmax(out_ad[0,:], dim=0)
                ad_prob = probs[1]
                ad_probs[idx] = ad_prob
        ad_pred = ad_probs.mean()
        return ad_pred



    def __getitem__(self, index) -> Tuple:

        # "split" language
        N = len(self.csv_data)
        idx = index % N
        age, gender, educ, ad, n_mmse = self.csv_data[idx]
        egemaps = self.ram_features[idx]
        features = (egemaps, torch.tensor(age), torch.tensor(gender), torch.tensor(educ))
        labels = (ad, n_mmse)

        if self.ad_models is not None:
            if self.ad_cache[idx] is not None:
                ad_prob = self.ad_cache[idx] # cached result
            else:
                ad_prob = self._get_ad_prob(features)
                self.ad_cache[idx] = ad_prob
            features = (*features, ad_prob)

        # GR sample injection in minibatches
        if self.do_rdm_greek:
            gr_N = len(self.csv_data_gr)
            gr_idx = index % gr_N
            gr_age, gr_gender, gr_educ, gr_ad, gr_n_mmse = self.csv_data_gr[gr_idx]
            gr_egemaps = self.ram_features_gr[gr_idx]
            gr_features = (gr_egemaps, torch.tensor(gr_age), torch.tensor(gr_gender), torch.tensor(gr_educ))
            gr_labels = (gr_ad, gr_n_mmse)
            if self.ad_models is not None:
                if self.gr_ad_cache[gr_idx] is not None:
                    ad_prob = self.gr_ad_cache[gr_idx] # cached result
                else:
                    ad_prob = self._get_ad_prob(features)
                    self.gr_ad_cache[gr_idx] = ad_prob
                gr_features = (*gr_features, ad_prob)

            if self.only_greek:
                return (gr_features, gr_labels)
            else:
                return (features, labels, gr_features, gr_labels)

        return (features, labels)
