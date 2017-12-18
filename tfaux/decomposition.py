import numpy as np
# from tensorly.base import unfold
# from tensorly.tenalg import multi_mode_dot

from .tensor import unfold, multi_mode_dot


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
    core : ndarray
        core tensor
        core.shape == tuple(ranks)
    factors : ndarray list
        list of factors
        factors[i].shape == (tensor.shape[i], ranks[i])
    """

    np.random.seed(random_state)
    core = np.random.random_sample(ranks)
    factors = [np.random.random_sample((tensor.shape[i], ranks[i]))
               for i in range(tensor.ndim)]
    while True:
        for mode in range(tensor.ndim):
            mode_list = list(range(tensor.ndim))
            mode_list.pop(mode)
            matrix_list = [factor for i, factor in enumerate(factors)
                           if i != mode]

            S_tensor = multi_mode_dot(tensor, matrix_list,
                                      modes=mode_list, transpose=True)
            S = unfold(S_tensor, mode)

            L = degrees[mode] - similarities[mode]

            # v : normalized eigenvectors
            _, v = np.linalg.eigh(np.dot(S, S.T) - alpha * L)

            r = factors[mode].shape[1]
            factors[mode] = v[:, -r:]

        print(loss_function_within_mode(tensor, core, factors,
                                        alpha, degrees, similarities))

    core = multi_mode_dot(tensor, factors, transpose=True)
    return core, factors


def _tucker_within_mode(tensor, degrees, similarities, n_iter_max,
                        alpha, tol, learning_rate, core, factors):
    """
    Tucker decompositon using graph laplacians with Within_Mode regularization.

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
    learning_rate : float
    core : ndarray
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

    for _ in range(n_iter_max):
        loss_prv = loss_function_within_mode(tensor, core, factors,
                                             alpha, degrees, similarities)
        # update factors
        for mode in range(tensor.ndim):
            # tensor_tmp = multi_mode_dot(core, factors)

            mode_list = list(range(tensor.ndim))
            mode_list.pop(mode)
            # matrix_list = [factor for i, factor in enumerate(factors) if i != mode]
            # S_tensor = multi_mode_dot(core, matrix_list, modes=mode_list)
            # L = degrees[mode] - similarities[mode]

            """
            slope = np.dot(unfold(tensor_tmp - tensor, mode),
            unfold(S_tensor, mode).T)
            slope += alpha * np.dot(L, factors[mode])

            factors[mode] -= learning_rate * slope
            """
            # S = unfold(S_tensor, mode)
            loss_crr = loss_function_within_mode(tensor, core, factors, alpha, degrees, similarities)

        # update core
        factors_tmp = [np.linalg.pinv(factor) for factor in factors]
        core = multi_mode_dot(tensor, factors_tmp)

        loss_crr = loss_function_within_mode(tensor, core, factors, alpha, degrees, similarities)
        if np.abs(loss_prv - loss_crr) / loss_crr < tol:
            return core, factors

    print('Could not obtain the convergence in tucker decompostion.')
    return core, factors


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


def tucker(tensor, ranks, similarities, n_iter_max=100, alpha=0.1, tol=1e-5, learning_rate=1e-2, regular='within',
           core=None, factors=None, random_state=None):
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
    learning_rate: float
    regular : {'within', 'cross'}
        a method for regularization
    core : {None, ndarray}
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
            import pdb; pdb.set_trace()
            raise ValueError(
                '{0}(dimention of similarities[{2}]) != {1}(dimention of mode {2} of tensor)'.format(
                    similarities[i].shape[0], dim, i))
    for similarity in similarities:
        if similarity.shape[0] != similarity.shape[1] or similarity.ndim != 2:
            raise ValueError('similarity should be a symetric matrix')

    # degree matrices
    degrees = [np.diag(np.sum(similarity, axis=0), k=0) for similarity in similarities]

    # init factors
    if (core is None) or (factors is None):
        core, factors = initial_core_factors(tensor, ranks, alpha, degrees, similarities, random_state)
    else:
        # TODO: write assertion for factors
        pass

    if regular == 'within':
        core, factors = _tucker_within_mode(tensor,
                                            degrees, similarities, n_iter_max, alpha, tol, learning_rate, core, factors)
    elif regular == 'cross':
        # TODO
        pass
    else:
        raise ValueError(
            'regularization method is `within` or `cross`, not {}'.format(regular))

    return core, factors
