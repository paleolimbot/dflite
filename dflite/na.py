
import numpy as np

NA = float("nan")
NA_character_ = "__NA_character_"


def is_nan(x):
    try:
        if type(x) == str:
            raise TypeError()
        return np.array([e == NA_character_ or (np.isreal(e) and np.isnan(e)) for e in x], dtype=bool)
    except TypeError:
        return x == NA_character_ or (np.isreal(x) and np.isnan(x))


def is_scalar(x):
    return np.ndim(x) == 0


def is_finite(x):
    try:
        if type(x) == str:
            return False
        return np.array([np.isreal(e) and np.isfinite(e) for e in x], dtype=bool)
    except TypeError:
        return np.isreal(x) and np.isfinite(x)
