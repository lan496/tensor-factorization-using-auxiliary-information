from warning import warn

import numpy as np
import tensorly


def _update_factor_within(tensor, laplacians, alpha, factors, mode):
    """
    Optimize loss function w.r.t. factors[mode]

    Parameters
    ----------
    tensor : ndarray
    laplacians : symmetric matrix list
        len(laplacians) == tensor.ndim
        laplacians[i].shape[0] == tensor.shape[i]
    alpha : float
        positive regularization constant
    factors : ndarray list
        list of factors
    mode : int

    Returns
    -------
    """

    mode_list = list(range(tensor.ndim))
    mode_list.pop(mode)
    matrix_list = [factor for i, factor in enumerate(factors) if i != mode]

    S_tensor = tensorly.tenalg.multi_mode_dot(tensor, matrix_list, modes=mode_list)
    S = tensorly.base.unfold(S_tensor, mode)

    # w : eigenvalues in ascending order
    # v : normalized eigenvectors
    w, v = np.linalg.eigh(np.dot(S, S.T) - alpha * laplacian[mode])

    r = factors[mode].shape[1]
    factors[mode] = v[:, -r:]


def _tucker_within_mode(tensor, laplacians, n_iter_max, tol, alpha, factors):
    """
    Tucker decompositon using graph laplacians with Within-Mode regularization.

    Parameters
    ----------
    tensor : ndarray
    laplacians : symmetric matrix list
        len(laplacians) == tensor.ndim
        laplacians[i].shape[0] == tensor.shape[i]
    n_iter_max : int
        maximum number of iteration
    tol : {None, float}
        tolerance
    alpha : float
        positive regularization constant
    regular : {'within', 'cross'}
        a method for regularization

    Returns
    -------
    core : ndarray
        core tensor
        core.shape == tuple(ranks)
    factors : ndarray list
        list of factors
        factors[i].shape == (tensor.shape[i], ranks[i])
    """

    for i in range(n_iter_max):
        for mode in range(tensor.ndim):
            _update_factor_within(tensor, laplacians, alpha, factors, mode)

        if tol:
            core_tmp = tensorly.tenalg.multi_mode_dot(tensor, [M.T for M in factors])
            tensor_tmp = tensorly.tucker_tensor.tucker_to_tensor(core_tmp, factors)
            err = np.sum(np.pow(tensor_tmp - tensor)) / np.sum(np.pow(tensor, 2))
            if err < tol:
                break

        if i == n_iter_max - 1 and tol:
            warn('Could not obtain the convergence in tucker decomposition.')

    return core, factors


def _tucker_cross_mode(tensor, laplacians, n_iter_max, tol, alpha, factors):
    pass


def tucker(tensor, ranks, laplacians, n_iter_max, alpha, tol=None, regular='within', random_state=None, factors=None):
    """
    Tucker decompositon using graph laplacians.

    Parameters
    ----------
    tensor : ndarray
    ranks : int list
        size of the core tensor
        len(ranks) == tensor.ndim
    laplacians : symmetric matrix list
        len(laplacians) == tensor.ndim
        laplacians[i].shape[0] == tensor.shape[i]
    n_iter_max : int
        maximum number of iteration
    alpha : float
        positive regularization constant
    tol : {None, float}
        tolerance
    regular : {'within', 'cross'}
        a method for regularization
    random_state : {None, int}

    Returns
    -------
    core : ndarray
        core tensor
        core.shape == tuple(ranks)
    factors : ndarray list
        list of factors
        factors[i].shape == (tensor.shape[i], ranks[i])
    """

    # check dimention
    if len(ranks) != tensor.ndim:
        raise ValueError(
            '{0}(length of ranks) != {1}(order of tensor)'.format(len(ranks), tensor.ndim))
    if len(laplacians) != tensor.ndim:
        raise ValueError(
            '{0}(length of laplacians) != {1}(order of tensor)'.format(len(laplacians), tensor.ndim))
    for i, dim in enumerate(tensor.shape):
        if laplacians[i].shape[0] != dim:
            raise ValueError(
                '{0}(dimention of laplacians[{2}]) != {1}(dimention of mode {2} of tensor)'.format(laplacians[i].shape[0], dim, i))
    for laplacian in laplacians:
        if laplacian.shape[0] != laplacian.shape[1] or laplacian.ndim != 2:
            raise ValueError('laplacian should be a symetric matrix')

    # init factors
    if not factors:
        np.random.seed(random_state)
        factors = [np.random.random_sample((tensor.shape[i], ranks[i]))
                   for i in range(tensor.ndim)]
    else:
        # TODO: write assertion for facrtors
        pass

    if regular == 'within':
        core, factors = _tucker_within_mode(tensor, laplacians, n_iter_max, tol, alpha, factors)
    elif regular == 'cross':
        core, factors = _tucker_cross_mode(tensor, laplacians, n_iter_max, tol, alpha, factors)
    else:
        raise ValueError(
            'regularization method is `within` or `cross`, not {}'.format(regular))

    return core, factors
