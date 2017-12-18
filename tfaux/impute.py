import numpy as np

from .tensor import multi_mode_dot
from .HOOI import tucker, loss_function_within_mode


def impute(tensor_with_nan, similarities, ranks, alpha, regular, tol, n_iter_max):
    fill_value = np.nanmean(tensor_with_nan)

    tensor = tensor_with_nan.copy()

    for index, e in np.ndenumerate(tensor):
        if np.isnan(e):
            tensor[index] = fill_value

    core, factors = None, None

    degrees = [np.diag(np.sum(similarity, axis=0), k=0) for similarity in similarities]

    loss_prv = None

    while True:
        for index, tr in np.ndenumerate(tensor_with_nan):
            if not np.isnan(tr):
                tensor[index] = tr

        core, factors = tucker(tensor, ranks, similarities, n_iter_max=n_iter_max, alpha=alpha, tol=tol,
                               regular=regular, factors=factors, random_state=0)

        loss_crr = loss_function_within_mode(tensor, core, factors, alpha, degrees, similarities)
        print(loss_crr)

        tensor = multi_mode_dot(core, factors)

        # if (loss_prv is not None) and (np.abs(loss_crr - loss_prv) / loss_prv < tol):
        if (loss_prv is not None) and np.isclose(loss_crr, loss_prv):
            break
        loss_prv = loss_crr

    return tensor
