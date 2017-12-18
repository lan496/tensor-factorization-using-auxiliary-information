import numpy as np

from .tensor import multi_mode_dot
from .decomposition import tucker


def impute(tensor_with_nan, similarities, ranks, alpha, regular, tol, n_iter_max, learning_rate):
    fill_value = np.nanmean(tensor_with_nan)

    tensor = tensor_with_nan.copy()

    for index, e in np.ndenumerate(tensor):
        if np.isnan(e):
            tensor[index] = fill_value

    core, factors = None, None

    while True:
        for index, tr in np.ndenumerate(tensor_with_nan):
            if not np.isnan(tr):
                tensor[index] = tr

        core, factors = tucker(tensor, ranks, similarities, n_iter_max=n_iter_max, alpha=alpha, tol=tol,
                               learning_rate=learning_rate, regular=regular, core=core, factors=factors, random_state=0)

        tensor_tmp = multi_mode_dot(core, factors)
        variation = np.sum(np.power(tensor_tmp - tensor, 2))
        print(variation)

        tensor = tensor_tmp

    return tensor
