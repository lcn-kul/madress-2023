from enum import Enum

# All data is seen X times per epoch to reduce number of checkpointed models.
TRAIN_TIMES_PER_EPOCH = 5
VAL_TIMES_PER_EPOCH = 1

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
    max_epochs=25,
    batch_size=32,
    default_lr=3e-4,
    default_warmup_steps=100,
    default_weight_decay=3e-2,
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
    ):

        # Save parameters.
        self.name = name
        self.label = label
        self.use_age = use_age
        self.use_gender = use_gender
        self.use_educ = use_educ
        self.use_ad_prob = use_ad_prob

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
        name=f"AD_CONFIG_{int(use_age)}{int(use_gender)}{int(use_educ)}",
        dim_hidden=12,
        label=Label.AD,
        use_age=use_age,
        use_gender=use_gender,
        use_educ=use_educ,
    )
    for use_age, use_gender, use_educ in [
        (False, False, False), # only audio
        (True, False, False), # audio + age
        (False, True, False), # audio + gender
        (False, False, True), # audio + educ
        (True, True, True), # audio + all covariates
    ]
]

MMSE_CONFIGS = [
    Config(
        name=f"MMSE_CONFIG_{int(use_age)}{int(use_gender)}{int(use_educ)}{int(use_ad_prob)}",
        dim_hidden=12,
        label=Label.MMSE,
        use_age=use_age,
        use_gender=use_gender,
        use_educ=use_educ,
        use_ad_prob=use_ad_prob,
    )
    for use_age, use_gender, use_educ, use_ad_prob in [
        (False, False, False, False), # only audio
        (True, False, False, False), # audio + age
        (False, False, True, False), # audio + educ
        (False, False, False, True), # audio + ad_prob
        (True, True, True, True), # audio + all covariates
    ]
]

# AD_CONFIG = Config(
#     name="AD_CONFIG",
#     dim_hidden=12,
#     label=Label.AD,
# )

# MMSE_CONFIG = Config(
#     name="MMSE_CONFIG",
#     dim_hidden=12,
#     label=Label.MMSE,
# )

FEAT_SEQ_LEN = 10
RDM_GREEK_EVERY = 5 # Randomly insert greek samples into the batches, every X samples