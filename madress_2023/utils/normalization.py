class NormTransform:
    def __init__(
        self,
        in_min: float,
        in_max: float,
        out_min: float,
        out_max: float,
        on_error="error",
    ):
        self.in_min = in_min
        self.in_max = in_max
        self.out_min = out_min
        self.out_max = out_max
        self.on_error = on_error

    @property
    def in_range(self) -> float:
        return self.in_max - self.in_min

    @property
    def out_range(self) -> float:
        return self.out_max - self.out_min

    def transform(self, x: float) -> float:
        frac = (x - self.in_min) / self.in_range
        return self.out_min + frac * self.out_range

    def transform_str(self, x: str, fmt="%0.6f") -> str:
        try:
            out = self.transform(float(x))
        except:
            if self.on_error == "error":
                raise
            elif self.on_error == "min":
                out = self.out_min
            elif self.on_error == "max":
                out = self.out_max
            elif self.on_error == "mid":
                out = (self.out_min + self.out_max) / 2
        return fmt % out


NORM_MMSE_TRANSFORM = NormTransform(
    in_min=0,
    in_max=30,
    out_min=0,
    out_max=1,
    on_error="mid",
)


def norm_ad_transform(x: str) -> int:
    if x == "ProbableAD":
        return 1
    elif x == "Control":
        return 0
    else:
        raise Exception(f"Unable to normalize AD: {x}")


def norm_education_transform(x: str) -> int:
    try:
        education = int(x)
    except:
        raise Exception(f"Unable to normalize education: {x}")
    return education


def norm_gender_transform(x: str) -> int:
    if x.lower() == "male":
        return 0
    if x.lower() == "female":
        return 1
    raise Exception(f"Unable to normalize gender: {x}")
