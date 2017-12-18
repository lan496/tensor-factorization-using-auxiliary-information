"""
Copyright (c) 2016 The tensorly Developers.
All rights reserved.
"""
import numpy as np


def unfold(tensor, mode):
    return np.reshape(np.moveaxis(tensor, mode, 0), (tensor.shape[mode], -1))


def fold(matrix, mode, shape):
    shape_list = list(shape)
    r = shape_list.pop(mode)
    shape_list.insert(0, r)
    return np.moveaxis(np.reshape(matrix, shape_list), 0, mode)


def mode_dot(tensor, matrix, mode):
    new_shape = list(tensor.shape)
    new_shape[mode] = matrix.shape[0]

    return fold(np.dot(matrix, unfold(tensor, mode)), mode, new_shape)


def multi_mode_dot(tensor, matrices_list, modes=None, transpose=False):
    if modes is None:
        modes = range(len(matrices_list))

    ret = tensor

    for mode, matrix in zip(modes, matrices_list):
        if transpose:
            ret = mode_dot(ret, np.transpose(matrix), mode)
        else:
            ret = mode_dot(ret, matrix, mode)

    return ret
