import numpy as np

from tucker import tucker


def impute(tensor_with_nan, ranks, alpha, regular, tol, interval):
    fill_value = 0

    tensor = tensor_with_nan.copy()

    for e in tensor.reshape(-1):
        e = fill_value

    tensor, factors = None, None

    while True:
        for tr, es in zip(tensor_with_nan.reshape(-1), tensor.reshape(-1)):
            if not np.isnan(tr):
                es = tr

        core, factors = tucker(tensor, ranks, laplacians, n_iter_max=interval, alpha=alpha, tol=None, regular=regular, random_state=None, factors=factors)
        tensor_tmp = tensorly.tucker_tensor.tucker_to_tensor(core, factors)

        variation = np.sum(np.pow(tensor_tmp - tensor, 2)) / np.sum(np.pow(tensor, 2))
        if variation < tol:
            break
        else:
            tensor = tensor_tmp

    return tensor
