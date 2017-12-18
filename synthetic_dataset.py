import numpy as np

from tfaux import impute, multi_mode_dot


def load_dataset(dim=30, R=2):
    U = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)
    V = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)
    W = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)

    core = np.zeros((R, R, R))
    for i in range(R):
        core[i, i, i] = 1.0

    tensor = multi_mode_dot(core, [U, V, W])

    A_I = np.diag([1, ] * (dim - 1), k=1) + np.diag([1, ] * (dim - 1), k=-1)
    A_J = A_I.copy()
    A_K = A_I.copy()

    similarities = [A_I, A_J, A_K]

    return tensor, similarities


def missing(tensor, missing_fraction):
    tensor_with_nan = tensor.copy()

    missing_number = int(missing_fraction * tensor.size)
    idx = np.random.choice(tensor.size, size=missing_number, replace=False)

    for i in idx:
        tensor_with_nan[np.unravel_index(i, tensor.shape)] = np.nan

    return tensor_with_nan


def MSE(tensor_true, tensor_with_nan, tensor_pred):
    mse = 0
    n = 0

    for index, e in np.ndenumerate(tensor_with_nan):
        if not np.isnan(e):
            continue
        mse += (tensor_true[index] - tensor_pred[index]) ** 2
        n += 1

    mse /= n
    return mse


def main():
    missing_fraction = 0.9

    ranks = [2, 2, 2]
    alpha = 0.1
    regular = 'within'
    tol = 1e-6
    n_iter_max = 100
    learning_rate = 1e-8

    tensor, similarities = load_dataset(dim=30)
    tensor_with_nan = missing(tensor, missing_fraction)

    tensor_estimate = impute(tensor_with_nan, similarities, ranks, alpha, regular, tol, n_iter_max, learning_rate)

    mse = MSE(tensor, tensor_with_nan, tensor_estimate)
    print('MSE: {}'.format(mse))


if __name__ == '__main__':
    main()
