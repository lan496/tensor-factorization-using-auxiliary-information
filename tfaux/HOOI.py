import numpy as np

from .tensor import multi_mode_dot, unfold


def initial_core_factors(tensor, ranks, alpha, degrees, similarities,
                         random_state=None):
    """
    Initialize core and factors for tucker decompostion.

    Parameters
    ----------
    tensor : ndarray
    ranks : int list
        size of the core tensor
    random_state : {None, int}

    Returns
    -------
    factors : ndarray list
        list of factors
        factors[i].shape == (tensor.shape[i], ranks[i])
    """

    np.random.seed(random_state)
    factors = [np.random.random_sample((tensor.shape[i], ranks[i]))
               for i in range(tensor.ndim)]
    return factors


def _update_factor_within(tensor, degrees, similarities, alpha, factors, mode):
    """
    Optimize loss function w.r.t. factors[mode]

    Parameters
    ----------
    tensor : ndarray
    degrees : symmetric matrix list
    similarities : symmetric matrix list
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

    S_tensor = multi_mode_dot(tensor, matrix_list, modes=mode_list, transpose=True)
    S = unfold(S_tensor, mode)
    L = degrees[mode] - similarities[mode]

    # v : normalized eigenvectors
    _, v = np.linalg.eigh(np.dot(S, S.T) - alpha * L)

    r = factors[mode].shape[1]
    factors[mode] = v[:, -r:]


def loss_function_within_mode(tensor, core, factors, alpha, degrees, similarities):
    """
    loss function for Within-Mode regularization

    Parameters
    ----------
    tensor : ndarray
    core : ndarray
    factors : {None, ndarray list}
    alpha : float
        positive regularization constant
    degrees : symmetric matrix list
    similarities : symmetric matrix list

    Returns
    -------
    loss : float
    """

    tensor_tmp = multi_mode_dot(core, factors)

    loss = 0.5 * np.sum(np.power(tensor - tensor_tmp, 2))
    for factor, degree, similarity in zip(factors, degrees, similarities):
        L = degree - similarity
        M = np.dot(factor.T, np.dot(L, factor))
        loss += 0.5 * alpha * np.trace(M)

    return loss


def _tucker_within_mode(tensor, degrees, similarities, n_iter_max, alpha, tol, factors):
    """
    Tucker decompositon using graph laplacians with Within-Mode regularization.

    Parameters
    ----------
    tensor : ndarray
    degrees : symmetric matrix list
    similarities : symmetric matrix list
    n_iter_max : int
        maximum number of iteration
    alpha : float
        positive regularization constant
    tol : {None, float}
        tolerance
    factors : ndarray list

    Returns
    -------
    core : ndarray
        core tensor
        core.shape == tuple(ranks)
    factors : ndarray list
        list of factors
        factors[i].shape == (tensor.shape[i], ranks[i])
    """

    core = multi_mode_dot(tensor, [M.T for M in factors])

    for i in range(n_iter_max):
        loss_prv = loss_function_within_mode(tensor, core, factors, alpha, degrees, similarities)

        # update factors
        for mode in range(tensor.ndim):
            _update_factor_within(tensor, degrees, similarities, alpha, factors, mode)

        core = multi_mode_dot(tensor, [M.T for M in factors])

        loss_crr = loss_function_within_mode(tensor,
                                             core, factors, alpha, degrees, similarities)
        if np.abs(loss_crr - loss_prv) / loss_prv < tol:
            return core, factors

    print('Could not obtain the convergence in tucker decomposition.')
    return core, factors


def _tucker_cross_mode(tensor, laplacians, n_iter_max, tol, alpha, factors):
    pass


def initial_factors(tensor, ranks):
    factors = []

    for i in range(tensor.ndim):
        X = unfold(tensor, i)
        _, v = np.linalg.eigh(np.dot(X, X.T))
        factor = v[:, -ranks[i]:]
        factors.append(factor)

    return factors


def tucker(tensor, ranks, similarities, n_iter_max=100, alpha=0.1, tol=1e-5, regular='within',
           factors=None, random_state=None):
    """
    Tucker decompositon using graph similarities.

    Parameters
    ----------
    tensor : ndarray
    ranks : int list
        size of the core tensor
        len(ranks) == tensor.ndim
    similarities : symmetric matrix list
        len(similarities) == tensor.ndim
        similarities[i].shape[0] == tensor.shape[i]
    n_iter_max : int
        maximum number of iteration
    alpha : float
        positive regularization constant
    tol : {None, float}
        tolerance
    regular : {'within', 'cross'}
        a method for regularization
    factors : {None, ndarray list}
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
    if len(similarities) != tensor.ndim:
        raise ValueError(
            '{0}(length of similarities) != {1}(order of tensor)'.format(len(similarities), tensor.ndim))
    for i, dim in enumerate(tensor.shape):
        if similarities[i].shape[0] != dim:
            raise ValueError(
                '{0}(dimention of similarities[{2}]) != {1}(dimention of mode {2} of tensor)'.format(
                    similarities[i].shape[0], dim, i))
    for similarity in similarities:
        if similarity.shape[0] != similarity.shape[1] or similarity.ndim != 2:
            raise ValueError('similarity should be a symetric matrix')

    # degree matrices
    degrees = [np.diag(np.sum(similarity, axis=0), k=0) for similarity in similarities]

    # init factors
    if factors is None:
        factors = initial_core_factors(tensor, ranks, alpha, degrees, similarities, random_state)
    else:
        # TODO: write assertion for factors
        pass

    if regular == 'within':
        core, factors = _tucker_within_mode(tensor,
                                            degrees, similarities, n_iter_max, alpha, tol, factors)
    elif regular == 'cross':
        # TODO
        pass
    else:
        raise ValueError(
            'regularization method is `within` or `cross`, not {}'.format(regular))

    return core, factors
