import numpy as np
from tensorly.tucker_tensor import tucker_to_tensor

from .tucker import tucker


def impute(tensor_with_nan, laplacians, ranks, alpha, regular, tol, interval):
    fill_value = np.nanmean(tensor_with_nan)

    tensor = tensor_with_nan.copy()

    for index, e in np.ndenumerate(tensor):
        if np.isnan(e):
            tensor[index] = fill_value

    factors = None

    while True:
        for index, tr in np.ndenumerate(tensor_with_nan):
            if not np.isnan(tr):
                tensor[index] = tr

        core, factors = tucker(tensor, ranks, laplacians, n_iter_max=interval, alpha=alpha, tol=None, regular=regular, factors=factors)

        tensor_tmp = tucker_to_tensor((core, factors))
        variation = np.sum(np.power(tensor_tmp - tensor, 2)) / np.sum(np.power(tensor, 2))
        print(variation)
        if variation < tol:
            break

        tensor = tensor_tmp

    return tensor
