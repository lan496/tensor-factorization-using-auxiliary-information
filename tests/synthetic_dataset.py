import numpy as np
import tensorly

# from tfaux import impute, tucker
from tfaux import impute, tucker


def load_dataset(dim=30, R=2):
    U = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)
    V = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)
    W = np.outer(np.array(np.arange(dim)), np.random.randn(R)) + np.random.randn(R)

    core = np.zeros((dim, dim, dim))
    for i in range(dim):
        core[i, i, i] = 1.0

    tensor = tensorly.tenalg.multi_mode_dot(core, [U, V, W])

    laplacian_I = -np.tri(dim) + 2 * np.eye(dim)
    laplacian_J = -np.tri(dim) + 2 * np.eye(dim)
    laplacian_K = -np.tri(dim) + 2 * np.eye(dim)

    laplacians = [laplacian_I, laplacian_J, laplacian_K]

    return tensor, laplacians


def missing(tensor, missing_fraction):
    tensor_with_nan = tensor.copy()

    missing_number = int(missing_fraction * tensor.size)
    idx = np.random.choice(tensor.size, size=missing_number, replace=False)

    for i in idx:
        tensor_with_nan.reshape(-1)[i] = np.nan

    return tensor_with_nan


def main():
    missing_fraction = 0.5

    ranks = [2, 2, 2]
    alpha = 0.1
    regular = 'within'
    tol = 0.001
    interval = 10

    tensor, laplacians = load_dataset()
    tensor_with_nan = missing(tensor, missing_fraction)

    tensor_estimate = impute(tensor_with_nan, ranks, alpha, regular, tol, interval)


if __name__ == '__main__':
    main()
