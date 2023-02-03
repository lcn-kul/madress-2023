import csv
import os
from pathlib import Path
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, ModelSummary, TQDMProgressBar
import torch
from torch.utils.data import DataLoader

from madress_2023 import constants
from madress_2023.constants import Split
from madress_2023.train.config import AD_CONFIGS, MMSE_CONFIGS, Config, TRAIN_ARGS, TRAIN_TIMES_PER_EPOCH, VAL_TIMES_PER_EPOCH, Label
from madress_2023.train.model_pl import ModelPL
from madress_2023.train.csv_dataset import CsvDataset
from madress_2023.train.extract_features import extract_features


def make_dataloader(
    split: Split,
    cpus: int,
    shuffle: bool,
    times_per_epoch: int = 1,
    do_rdm_greek: bool = False,
    fold = None,
    only_greek: bool = False,
    ad_model = None,
):

    # Create DataLoader.
    csv_dataset = CsvDataset(
        split,
        times_per_epoch,
        do_rdm_greek,
        fold=fold,
        only_greek=only_greek,
        ad_model=ad_model,
    )
    csv_dataloader = DataLoader(
        csv_dataset,
        batch_size=TRAIN_ARGS.batch_size,
        shuffle=shuffle,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )
    return csv_dataloader, csv_dataset


def _get_last_checkpoint(model_dir):
    ckpt_paths = list(model_dir.glob("all*.ckpt"))
    if len(ckpt_paths) > 0:
        # Find max epoch path .
        stems = [os.path.splitext(os.path.basename(str(x)))[0] for x in ckpt_paths]
        parts_per_stem = [x.split("-") for x in stems]
        dict_per_stem = [{p.split("=")[0]: p.split("=")[1]
                        for p in parts if "=" in p} for parts in parts_per_stem]
        epoch_per_stem = [x["epoch"] for x in dict_per_stem]
        max_idx = max(range(len(epoch_per_stem)),
                        key=epoch_per_stem.__getitem__)
        ckpt_path = ckpt_paths[max_idx]
    else:
        ckpt_path = None
    return ckpt_path

def _load_best_model(model: ModelPL, model_dir: Path):
    all_models = []
    ckpt_paths = list(model_dir.glob("all*.ckpt"))
    if len(ckpt_paths) == 0:
        print("NO CHECKPOINTS, RETURNING MODEL")
        return model
    
    for ckpt_path in sorted(ckpt_paths):
        all_models.append(ModelPL.load_from_checkpoint(ckpt_path))
    
    # Load losses csv.
    losses = []
    with open(model_dir.joinpath("losses.csv"), mode="r") as f:
        csv_reader = csv.reader(f)
        for idx, row in enumerate(csv_reader):
            if idx == 0:
                continue
            epoch,loop,_,loss = row
            if loop == "train":
                continue
            losses.append((int(epoch), float(loss)))

    # Find best loss.
    min_loss = None
    best_epoch = None
    for epoch, loss in losses:
        if min_loss is None or loss < min_loss:
            min_loss = loss
            best_epoch = epoch
    best_model: ModelPL = all_models[best_epoch]
    model.load_state_dict(best_model.state_dict())

    return model

def _load_best_pretrain_model(model: ModelPL, model_dirs: Path):
    losses = []

    all_models = []
    for model_dir in model_dirs:
        ckpt_paths = list(model_dir.glob("all*.ckpt"))
        if len(ckpt_paths) == 0:
            print("NO CHECKPOINTS, RETURNING MODEL")
            return model
        
        for ckpt_path in sorted(ckpt_paths):
            all_models.append(ModelPL.load_from_checkpoint(ckpt_path))
        
        # Load losses csv.
        with open(model_dir.joinpath("losses.csv"), mode="r") as f:
            csv_reader = csv.reader(f)
            for idx, row in enumerate(csv_reader):
                if idx == 0:
                    continue
                epoch,loop,_,loss = row
                if loop == "train":
                    continue
                _ii = len(losses)
                losses.append((_ii, int(epoch), float(loss)))

    # Find best loss per config.
    min_loss = None
    best_model_idx = None
    for model_idx, epoch, loss in losses:
        if min_loss is None or loss < min_loss:
            min_loss = loss
            best_model_idx = model_idx
    best_model: ModelPL = all_models[best_model_idx]
    model.load_state_dict(best_model.state_dict())

    return model


def _model_averaging(model_dir: Path, config: Config):
    print("model averaging")

    # Initial model.
    _out_csv_path = model_dir / "losses.csv"
    model = ModelPL(config, _out_csv_path)

    # Do averaging (or load from file if it already exists).
    out_path = model_dir.joinpath("avg_model.pt")
    if out_path.exists():
        state_dict = torch.load(str(out_path))
    else:
        fold_models = []
        num_folds = 2
        for i in range(num_folds):
            fold_out_name = f"trained_model_{config.name}_fold_{i}"
            fold_model_dir = constants.DIR_MODELS.joinpath(fold_out_name)
            fold_out_csv_path = fold_model_dir / "losses.csv"
            _model = ModelPL(config, fold_out_csv_path)
            _model = _load_best_model(_model, fold_model_dir)
            fold_models.append(_model)

        fold_state_dicts = [x.state_dict() for x in fold_models]
        state_dict = model.state_dict()
        for key in state_dict:
            state_dict[key] = sum(fold_state_dicts[i][key] for i in range(num_folds)) / float(num_folds)
        os.makedirs(str(out_path.parent), mode=0o755, exist_ok=True)
        torch.save(state_dict, str(out_path))
    model.load_state_dict(state_dict)
    return model

def train_model(
    train_split: Split,
    val_split: Split,
    test_split: Split,
    cpus: int,
    fold: int = None,
    pretrain_idx: int = None,
    num_pretrain: int = None,
    config: Config = None,
):
    seed = 42
    if pretrain_idx is not None:
        seed += (config.label.value*num_pretrain) + pretrain_idx
        # seed += pretrain_idx
    pl.seed_everything(seed)

    if cpus > 5: # avoid having too many cache instances over multiple processes
        cpus = 5

    do_rdm_greek = (fold is not None)

    # If something went wrong and we need to restart the job pipeline...
    print("Calculating features for this training iteration", flush=True)
    extract_features(train_split, fold)
    extract_features(val_split, fold)
    extract_features(test_split, fold)
    if do_rdm_greek:
        gr_train_split = Split.GR_SAMPLE_2FOLDS_TRAIN
        gr_val_split = Split.GR_SAMPLE_2FOLDS_VAL
        extract_features(gr_train_split, fold)
        extract_features(gr_val_split, fold)

    if config.label == Label.AD: # FIRST TRAIN AD MODEL
        ad_model = None
    else: # THEN TRAIN MMSE MODEL USING AD PREDICTIONS
        ad_avg_dir = constants.DIR_MODELS.joinpath("trained_model_AD_CONFIG_111_avg")
        ad_model = _model_averaging(ad_avg_dir, AD_CONFIGS[-1]).to("cpu")


    # Trainer parameters.
    if pretrain_idx is not None:
        out_name = f"trained_model_{config.name}_pretrain_{pretrain_idx}"
    elif fold is not None:
        out_name = f"trained_model_{config.name}_fold_{fold}"
    model_dir = constants.DIR_MODELS.joinpath(out_name)

    # Create model.
    _out_csv_path = model_dir / "losses.csv"
    model = ModelPL(config, _out_csv_path)

    # Create dataloader(s).
    train_dl, train_ds = make_dataloader(
        train_split,
        cpus,
        shuffle=True,
        times_per_epoch=TRAIN_TIMES_PER_EPOCH,
        do_rdm_greek=do_rdm_greek,
        fold=fold,
        ad_model=ad_model,
    )
    val_dl, val_ds = make_dataloader(
        val_split,
        cpus,
        shuffle=False,
        times_per_epoch=VAL_TIMES_PER_EPOCH,
        do_rdm_greek=do_rdm_greek,
        fold=fold,
        only_greek=do_rdm_greek, # validate on greek samples only
        ad_model=ad_model,
    )


    all_ckpt_callback = ModelCheckpoint(
        dirpath=str(model_dir),
        filename="all-{epoch:03d}",
        every_n_epochs=1,
        save_top_k=-1,
    )
    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    summary_callback = ModelSummary(max_depth=4)
    tb_logger = TensorBoardLogger(save_dir=str(constants.DIR_LOGS / out_name))

    all_callbacks = [all_ckpt_callback, progress_bar_callback, summary_callback]


    # Device for model computations.
    if torch.cuda.is_available():
        gpus = 1
        device = "cuda"
    else:
        gpus = 0
        device = "cpu"
    print(f"Using: %s" % device)

    trainer_params = {
        "gpus": gpus,
        "max_epochs": TRAIN_ARGS.max_epochs,
        "callbacks": all_callbacks,
        "enable_progress_bar": True,
        "num_sanity_val_steps": 2,
        "log_every_n_steps": 10,
        "logger": tb_logger,
        "deterministic": True,
    }
    trainer = pl.Trainer(**trainer_params)

    ckpt_path = _get_last_checkpoint(model_dir)

    # start from pretrained english model for fold finetuning, this way we can do
    # model weight averaging in the end
    if ckpt_path is None and fold is not None:
        pretrain_out_names = [f"trained_model_{config.name}_pretrain_{i}" for i in range(num_pretrain)]
        pretrained_model_dirs = [constants.DIR_MODELS.joinpath(x) for x in pretrain_out_names]
        model = _load_best_pretrain_model(model, pretrained_model_dirs)

    if ckpt_path is not None:
        trainer.fit(model, train_dl, val_dl, ckpt_path=str(ckpt_path))
    else:
        trainer.fit(model, train_dl, val_dl)
    
    train_ds.times_per_epoch = 1
    val_ds.times_per_epoch = 1
    eval_dl_train = DataLoader(
        train_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )
    eval_dl_val = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )

    progress_bar_callback = TQDMProgressBar(refresh_rate=10)
    summary_callback = ModelSummary(max_depth=4)

    all_callbacks = [progress_bar_callback, summary_callback]


    # Device for model computations.
    if torch.cuda.is_available():
        gpus = 1
        device = "cuda"
    else:
        gpus = 0
        device = "cpu"
    print(f"Using: %s" % device)

    trainer_params = {
        "gpus": gpus,
        "callbacks": all_callbacks,
        "enable_progress_bar": True,
        "logger": False,
    }
    trainer = pl.Trainer(**trainer_params)

    model.eval()
    print("EVALUATING: train_ds")
    trainer.test(model, eval_dl_train, ckpt_path=None)
    print("EVALUATING: val_ds")
    trainer.test(model, eval_dl_val, ckpt_path=None)

    print("PREDICTIONG: test_ds")
    test_ds = CsvDataset(
        test_split,
        ad_model=ad_model,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )

    trainer.predict(model, test_dl, ckpt_path=None)

def averaging(config: Config, test_split: Split):
    if config.label == Label.AD:
        # Phase 1: train AD model
        ad_model = None
    else: 
        # Phase 2: use AD model predictions when training MMSE model
        ad_avg_dir = constants.DIR_MODELS.joinpath("trained_model_AD_CONFIG_111_avg")
        ad_model = _model_averaging(ad_avg_dir, AD_CONFIGS[-1]).to("cpu")

    # model averaging
    avg_dir = constants.DIR_MODELS.joinpath(f"trained_model_{config.name}_avg")
    model = _model_averaging(avg_dir, config)


    print("PREDICTIONG: test_ds")
    trainer = pl.Trainer(logger=False)
    # Create DataLoader.
    test_ds = CsvDataset(
        test_split,
        ad_model=ad_model,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cpus-1,
        persistent_workers=(cpus > 1),
    )
    model.eval()
    trainer.predict(model, test_dl, ckpt_path=None)




if __name__ == "__main__":
    pretrain_train_split = Split.EN_BALANCED_TRAIN
    pretrain_val_split = Split.GR_SAMPLE
    fold_train_split = Split.EN_BALANCED_TRAIN # with rdm_greek in batches
    fold_val_split = Split.EN_BALANCED_VAL # with rdm_greek in batches
    test_split = Split.GR_TEST
    num_folds = 2
    cpus: int = 1
    num_pretrain = 5

    for config in AD_CONFIGS + MMSE_CONFIGS:

        # pretraining english
        for i in range(num_pretrain):
            train_model(
                pretrain_train_split,
                pretrain_val_split,
                test_split,
                cpus,
                pretrain_idx=i,
                num_pretrain=num_pretrain,
                config=config,
            )

        # finetuning on folds
        for i in range(num_folds):
            train_model(
                fold_train_split,
                fold_val_split,
                test_split,
                cpus,
                fold=i,
                config=config,
                num_pretrain=num_pretrain,
            )

        # model averaging
        averaging(config, test_split)
