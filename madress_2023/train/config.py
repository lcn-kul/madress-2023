from enum import Enum

# All data is seen X times per epoch to reduce number of checkpointed models.
TRAIN_TIMES_PER_EPOCH = 1
VAL_TIMES_PER_EPOCH = 1

FEAT_SEQ_LEN = 10
RDM_GREEK_EVERY = 5 # Randomly insert greek samples into the batches, every X samples

class Label(Enum):
    AD = 0
    MMSE = 1


class TrainConfig:

    max_epochs: int = None
    batch_size: int = None
    default_lr: float = None
    default_warmup_steps: int = None
    default_weight_decay: float = None

    def __init__(
        self,
        max_epochs: int,
        batch_size: int,
        default_lr: float,
        default_warmup_steps: int,
        default_weight_decay: float,
    ) -> None:
        assert max_epochs > 0
        assert batch_size > 0
        assert default_lr > 0
        assert default_warmup_steps >= 0
        assert default_weight_decay > 0

        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.default_lr = default_lr
        self.default_warmup_steps = default_warmup_steps
        self.default_weight_decay = default_weight_decay


TRAIN_ARGS = TrainConfig(
    max_epochs=30,
    batch_size=32,
    default_lr=3e-3,
    default_warmup_steps=100,
    default_weight_decay=1e-2,
)


class Config:
    def __init__(
        self,
        name: str,
        dim_hidden: int,
        label: Label = None,
        use_age: bool = False,
        use_gender: bool = False,
        use_educ: bool = False,
        use_ad_prob: bool = False, # only relevant for MMSE
        seed_idx: int = 0, # different seeds for submission
    ):

        # Save parameters.
        self.name = name
        self.label = label
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_educ = use_educ
        self.use_ad_prob = use_ad_prob
        self.seed_idx = seed_idx

        self.do_ad = label == Label.AD
        self.do_mmse = label == Label.MMSE

        # input dimension
        _h = 25  # eGeMAPS-25 (low-level descriptors)

        # covariates (age, gender, eduction, ad_prob)
        _num_covars = 0
        if use_age:
            _num_covars += 1
        if use_gender:
            _num_covars += 1
        if use_educ:
            _num_covars += 1
        if label == Label.MMSE and use_ad_prob:
            _num_covars += 1

        self.dim_input = _h + _num_covars
        self.dim_hidden = dim_hidden
        self.dropout = 0.2

AD_CONFIGS = [
    Config(
        name=f"AD_CONFIG_SEED{i}",
        dim_hidden=12,
        label=Label.AD,
        use_age=True,
        use_gender=True,
        use_educ=True,
        seed_idx=i
    )
    for i in range(5)
]

MMSE_CONFIGS = [
    Config(
        name=f"MMSE_CONFIG_SEED{i}",
        dim_hidden=8,
        label=Label.MMSE,
        use_age=True,
        use_gender=True,
        use_educ=True,
        use_ad_prob=True,
        seed_idx=i
    )
    for i in range(5)
]
