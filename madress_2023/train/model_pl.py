from pathlib import Path
import pytorch_lightning as pl
import random
import torch
from torch.nn import functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from madress_2023.train.config import Config, TRAIN_ARGS, RDM_GREEK_EVERY, Label
from madress_2023.train.model import Model


class ModelPL(pl.LightningModule):

    def __init__(
        self,
        config: Config,
        out_csv_path: Path = None,
    ):
        super().__init__()
        self.config = config
        self.label_str = config.label.name
        self.greek_every = RDM_GREEK_EVERY
        self.train_args = TRAIN_ARGS

        self.model = Model(config)

        # Needed to configure learning rate.
        # See: training_step()
        self.automatic_optimization = False

        if out_csv_path is not None:
            self.out_csv_path = out_csv_path
            self.out_test_path = out_csv_path.parent.joinpath("metric.txt")
            self.predict_path = out_csv_path.parent.joinpath("prediction.txt")

            # Create CSV if it doesn't exist. We will append it every epoch.
            HEADER = ["epoch", "loop", "label", "loss"]
            if not out_csv_path.exists():
                out_csv_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
                with open(out_csv_path, mode="w") as f:
                    f.write(",".join(HEADER) + "\n")

            # Clear out_test file.
            with open(self.out_test_path, mode="w") as f:
                f.write("")
            
        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=self.model.parameters(),
            lr=self.train_args.default_lr,
            weight_decay=self.train_args.default_weight_decay,
        )
        return [optimizer]

    def on_train_epoch_start(self) -> None:
        self.train_losses = []
        self.val_losses = []

    def _preprocess_batch(self, batch):
        # Batch will either be
        # - (feats, labels)
        # - (feats, labeles, feats_rdm_gr, labels_rdm_gr)
        if len(batch) == 2: # feats, labels
            do_rdm_greek = False
        elif len(batch) == 4: # feats, labels, feats_gr, labels_gr
            do_rdm_greek = True

        if do_rdm_greek:
            offset = random.randrange(self.greek_every) # random offset
            features, labels, gr_features, gr_labels = batch
            for i in range(len(features)):
                features[i][offset::self.greek_every] = gr_features[i][offset::self.greek_every]
            for i in range(len(labels)):
                labels[i][offset::self.greek_every] = gr_labels[i][offset::self.greek_every]
        else:
            features, labels = batch

        # Attach covariates (age, gender, education) to acoustic/linguistic features.
        # If MMSE, then ad_prob will also be in the list of "covariates"
        _egemaps, *covars = features
        
        # Select covars based on config (age, gender, educ, ad_prob)
        selected_covars = []
        if self.config.use_age:
            selected_covars.append(covars[0])
        if self.config.use_gender:
            selected_covars.append(covars[1])
        if self.config.use_educ:
            selected_covars.append(covars[2])
        if self.config.use_ad_prob:
            selected_covars.append(covars[3])
        if len(selected_covars) > 0:
            if selected_covars[0].dim() == 0:
                _view_shape = (1,1,)
            else:
                _view_shape = (selected_covars[0].shape[-1], 1,1)
            _covars = tuple(
                covar.view(*_view_shape).expand((*_egemaps.shape[:-1], 1))
                for covar in selected_covars
            )
            new_features = torch.cat((_egemaps, *_covars), dim=-1)
        else:
            new_features = _egemaps
        return new_features, labels

    def _forward_loss(self, features, labels, backward: bool = False, opt_step: bool = False, no_grad = False):
        if no_grad:
            with torch.no_grad():
                out = self.model.forward(features)
        else:
            out = self.model.forward(features)
        if self.config.do_ad:
            loss = F.cross_entropy(out, labels[0])
        if self.config.do_mmse:
            loss = F.mse_loss(out, labels[1])
        if backward:
            loss.backward()
        if opt_step:
            self.optimizers().step()
            self.optimizers().zero_grad()
        return loss

    def training_step(self, train_batch, batch_idx):
        features, labels = self._preprocess_batch(train_batch)

        # Warmup implementation, scale lr down with factor derived from
        # linear warmup. Based on:
        # https://github.com/Lightning-AI/lightning/issues/328#issuecomment-550114178
        _lr = self.train_args.default_lr
        warmup_steps = self.train_args.default_warmup_steps
        cur_steps = self.trainer.global_step
        if cur_steps < warmup_steps:
            lr_scale = float(cur_steps + 1) / warmup_steps
            _lr *= lr_scale
            opt = self.optimizers()
            for pg in opt.param_groups:
                pg['lr'] = _lr

        loss = self._forward_loss(features, labels, backward=True, opt_step=True)
        loss_cpu = loss.detach().cpu()
        self.train_losses.append(loss_cpu)

        self.log("lr", torch.tensor(_lr))
        self.log("eff_step", torch.tensor(cur_steps))
        self.log(f"train_loss_{self.label_str}", loss_cpu)


    def on_train_epoch_end(self) -> None:
        _losses = torch.stack(self.train_losses, dim=0)
        _mean_loss = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            _epoch = "%i" % self.current_epoch
            _loss = "%0.8f" % _mean_loss.item()
            row = [_epoch, "train", self.label_str, _loss]
            f.write(",".join(row) + "\n")

    def on_validation_start(self) -> None:
        self.val_losses = []

    def validation_step(self, val_batch, batch_idx):
        features, labels = self._preprocess_batch(val_batch)

        if self.trainer.sanity_checking:
            self.val_losses = []

        loss = self._forward_loss(features, labels, no_grad=True)
        loss_cpu = loss.detach().cpu()
        self.val_losses.append(loss_cpu)
        self.log(f"val_loss_{self.label_str}", loss_cpu)

    def on_validation_epoch_end(self) -> None:
        # sanity check will enter here
        if self.trainer.sanity_checking:
            print("Skipping sanity check write to CSV")
            return
        _losses = torch.stack(self.val_losses, dim=0)
        _mean_loss = _losses.mean(dim=0)

        with open(self.out_csv_path, mode="a") as f:
            _epoch = "%i" % self.current_epoch
            _loss = "%0.8f" % _mean_loss.item()
            row = [_epoch, "val", self.label_str, _loss]
            f.write(",".join(row) + "\n")

    def on_test_epoch_start(self) -> None:
        num_examples = len(self.trainer.test_dataloaders[0].dataset)
        self.test_ad_probs = torch.zeros((num_examples,))
        self.test_ad_labels = torch.zeros((num_examples,), dtype=torch.int64)
        self.test_num_correct = torch.zeros((1,))
        self.test_squared_err = torch.zeros((1,))
        self.test_denorm_squared_err = torch.zeros((1,))
        self.test_samples = 0

    def test_step(self, test_batch, batch_idx):
        features, labels = self._preprocess_batch(test_batch)
        with torch.no_grad():
            out = self.model.forward(features)
        if self.config.do_ad:
            probs = F.softmax(out[0, :], dim=0)
            ad_prob = probs[1].item()
        if self.config.do_mmse:
            mmse = out[0].item()

        ad_label = labels[0][0].item()
        mmse_label = labels[1][0].item()

        
        if self.config.do_ad:
            self.test_ad_probs[self.test_samples] = ad_prob
            _ad_tn = ad_prob < 0.5 and ad_label == 0
            _ad_tp = ad_prob >= 0.5 and ad_label == 1
            _ad_correct = _ad_tn or _ad_tp
            if _ad_correct:
                self.test_num_correct += 1
        if self.config.do_mmse:
            self.test_squared_err += (mmse_label - mmse) ** 2
            self.test_denorm_squared_err += (30.0*mmse_label - 30.0*mmse) ** 2

        self.test_ad_labels[self.test_samples] = labels[0][0].item()
        self.test_samples += 1


    def on_test_epoch_end(self) -> None:

        # Also write console log to file.
        result_strs = []

        # Calculate accuracy/MSE.
        if self.config.do_ad:
            acc = self.test_num_correct / self.test_samples
            acc_str = "%0.2f %%" % (acc.item()*100.0)
            _msg = f"{self.config.name}: acc AD {acc_str}"
            print(_msg)
            result_strs.append(_msg)
        if self.config.do_mmse:
            mse = self.test_squared_err / self.test_samples
            denorm_mse = self.test_denorm_squared_err / self.test_samples
            rmse = mse.sqrt()
            denorm_rmse = denorm_mse.sqrt()
            rmse_str = "%0.3f" % rmse.item()
            denorm_rmse_str = "%0.3f" % denorm_rmse.item()
            _msg = f"{self.config.name}: RMSE MMSE {rmse_str} / denorm = {denorm_rmse_str}"
            print(_msg)
            result_strs.append(_msg)

        with open(self.out_test_path, mode="a") as f:
            for _msg in result_strs:
                f.write(_msg + "\n")


    def on_predict_start(self) -> None:
        self.predictions = []

    def predict_step(self, predict_batch, predict_idx):
        features, _ = self._preprocess_batch(predict_batch)
        with torch.no_grad():
            out = self.model.forward(features)
        if self.config.do_ad:
            probs = F.softmax(out[0, :], dim=0)
            ad_prob = probs[1].item()
            if ad_prob >= 0.5:
                self.predictions.append(1)
            else:
                self.predictions.append(0)
        if self.config.do_mmse:
            mmse = out[0].item()
            self.predictions.append(mmse)


    def on_predict_end(self) -> None:
        predict_strs = []
        for value in self.predictions:
            if self.config.do_ad:
                predict_strs.append("ProbableAD" if value == 1 else "Control")
            elif self.config.do_mmse:
                predict_strs.append("%0.5f" % (value*30.))

        with open(self.predict_path, mode="w") as f:
            for _str in predict_strs:
                f.write(_str + "\n")

