# -*- coding: utf-8 -*-
# MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
#
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
import collections
from typing import Iterable, Union

import numpy as np

from .._imperative_rt import make_const
from .._imperative_rt.core2 import SymbolVar, Tensor, apply, dtype_promotion, get_device
from .._wrap import as_device
from ..ops import builtin
from ..ops.special import Const
from .amp import _high_prec_dtype, _low_prec_dtype
from .dtype import is_dtype_equal, is_quantize

_enable_convert_inputs = True


def get_convert_inputs():
    """ get the curerent state of `_enable_convert_inputs` """
    return _enable_convert_inputs


def set_convert_inputs(flag):
    """ This function is a temporary workaround for reducing the overhead of operator
    invocations. The function `convert_inputs` is disabled if the global state
    `_enable_convert_inputs` is set to `False`, otherwise enabled. This function is for
    internal use only, and should be removed when the tensor-like system is refactored.
    """
    global _enable_convert_inputs
    backup = _enable_convert_inputs
    _enable_convert_inputs = flag
    return backup


def concatenate(inputs, axis=0, *, device=None):
    inputs = convert_inputs(*inputs)
    if device is None:
        device = get_device(inputs)
    (result,) = apply(builtin.Concat(axis=axis, comp_node=device), *inputs)
    return result


def astype(x, dtype):
    dtype = np.dtype(dtype)
    if not is_dtype_equal(x.dtype, dtype):
        isscalar = x._isscalar()
        (x,) = apply(builtin.TypeCvt(dtype=dtype), x)
        if isscalar:
            x._setscalar()
    return x


def convert_single_value(v, *, dtype=None, device=None):
    if isinstance(v, (Tensor, SymbolVar)):
        if not is_quantize(v.dtype):
            v = astype(v, dtype)
    else:
        (v,) = Const(v, dtype=dtype, device=device)()
    return v


def convert_inputs(*args, device=None):
    if not _enable_convert_inputs:
        return args

    dtype = dtype_promotion(args)
    if device is None:
        device = get_device(args)
    device = as_device(device)

    graph = None
    sym_type = None
    for a in args:
        if isinstance(a, SymbolVar):
            if graph is None:
                graph = a.var.graph
                sym_type = type(a)
            else:
                assert graph == a.var.graph
    args = list(args)
    if graph is not None:
        for i in range(len(args)):
            if not isinstance(args[i], SymbolVar):
                rst = make_const(graph, np.array(args[i]), device.to_c(), dtype)
                args[i] = sym_type(rst)

    def convert(value):
        if value is None:
            return value
        return convert_single_value(value, dtype=dtype, device=device.to_c())

    return tuple(map(convert, args))


def cast_tensors(*args, promote=False):
    if promote:
        dtype = _high_prec_dtype
    else:
        dtype = _low_prec_dtype
    return tuple(arg.astype(dtype) if arg is not None else None for arg in args)


def result_type(*args):
    dtypes = []
    for i in args:
        if isinstance(i, Tensor):
            dtypes.append(i.dtype)
            continue
        try:
            dtypes.append(np.dtype(i))
        except TypeError:
            pass
    return np.result_type(*dtypes)


def isscalar(x):

    if isinstance(x, (Tensor, SymbolVar)):
        return x._isscalar()

    return np.isscalar(x)


def setscalar(x):
    if isinstance(x, (Tensor, SymbolVar)):
        x._setscalar()
    else:
        raise NotImplementedError("Unsupport type {}".format(type(x)))


def astensor1d(x, *reference, dtype=None, device=None):
    """
    Convert something to 1D tensor. Support following types
    * sequence of scalar literal / tensor
    * numpy array
    * tensor (returned as is, regardless of dtype and device)
    """
    try:
        ndim = x.ndim
    except AttributeError:
        pass
    except ValueError:
        if dtype is not None and dtype != x.dtype:
            x = astype(x, dtype)
        if device is not None:
            cn = as_device(device).to_c()
            (x,) = apply(builtin.Copy(comp_node=cn), x)
        return x
    else:
        if ndim != 0 and ndim != 1:
            raise ValueError("ndim != 1 or 0, get : %d" % ndim)
        if not isinstance(x, (Tensor, SymbolVar)):
            (x,) = Const(x, dtype=dtype, device=device)(*reference)
        return x

    if not isinstance(x, collections.abc.Sequence):
        raise TypeError

    if any(isinstance(i, (Tensor, SymbolVar)) for i in x):
        x = concatenate(x, device=device) if len(x) > 1 else x[0]
        if dtype is not None:
            x = astype(x, dtype)
        return x
    (x,) = Const(x, dtype=dtype, device=device)(*reference)
    return x


def _expand_int(s, i):
    if isinstance(i, (Tensor, SymbolVar)):
        i_np = i.numpy()
        if i_np.ndim == 0:
            s.append(int(i_np))
        else:
            s += list(i_np)
        return
    if isinstance(i, Iterable):
        for ii in i:
            _expand_int(s, ii)
        return
    if np.issubdtype(type(i), np.integer):
        s.append(i)
        return
    raise


def make_shape_tuple(shape):
    s = []
    _expand_int(s, shape)
    return tuple(s)


def _normalize_axis(
    ndim: int, axis: Union[int, Iterable], reverse=False
) -> Union[int, list]:
    def convert(x):
        x_org = x
        if x < 0:
            x = ndim + x
        assert (
            x >= 0 and x < ndim
        ), "axis {} is out of bounds for tensor of dimension {}".format(x_org, ndim)
        return x

    if isinstance(axis, int):
        return convert(axis)
    elif isinstance(axis, Iterable):
        axis_org = axis
        axis = list(sorted(map(convert, axis), reverse=reverse))
        for i in range(len(axis) - 1):
            assert axis[i] != axis[i + 1], "axis {} contains duplicated indices".format(
                axis_org
            )
        return axis
    raise
