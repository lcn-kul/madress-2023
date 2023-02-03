from madress_2023 import constants


def full_path(path: str, make_dir: bool = True):
    full_path = constants.DIR_PROJECT.joinpath(path).resolve()
    if make_dir:
        try:
            full_path.parent.mkdir(mode=0o755, parents=True, exist_ok=True)
        except:
            pass
    return str(full_path)
