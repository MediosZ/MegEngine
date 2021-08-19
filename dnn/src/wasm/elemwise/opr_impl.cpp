/**
 * \file dnn/src/wasm/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/wasm/elemwise/opr_impl.h"
#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <iostream>

using namespace megdnn;
using namespace wasm;

bool ElemwiseImpl::exec_unary() {
    if (m_src->size() != 1)
        return false;

    // some optr only takes input data of float_32
    if (m_dst->layout.dtype != dtype::Float32())
        return false;

    auto elparam = make_elemwise_op_param<1>();
    if (!elparam[0].layout.is_contiguous())
        return false;
    megdnn_assert(elparam[0].layout.ndim == 1);

#define DISPATCH_XNNPACK(_mode, _func)                      \
    case Mode::_mode:                                       \
        MEGDNN_DISPATCH_CPU_KERN_OPR(_func(sptr, dptr, n)); \
        return true

    auto n = elparam[0].layout.shape[0];
    auto sptr = elparam[0].ptr<dt_float32>(),
            dptr = m_dst->ptr<dt_float32>();

    auto xnnpack_dispatch = [&]() {
        switch (param().mode) {
            DISPATCH_XNNPACK(ABS, xnnAbs);
            DISPATCH_XNNPACK(FLOOR, xnnFloor);
            DISPATCH_XNNPACK(CEIL, xnnCeil);
            DISPATCH_XNNPACK(SIGMOID, xnnSigmoid);
            // round fails
            // DISPATCH_XNNPACK(ROUND, xnnRound);
            DISPATCH_XNNPACK(RELU, xnnRelu);
            default:
                return false;
        }
    };
    return xnnpack_dispatch();
#undef DISPATCH_XNNPACK

    return false;
}

bool ElemwiseImpl::exec_binary() {
    if (m_src->size() != 2 ||
        m_src->front().layout.dtype != m_dst->layout.dtype ||
        m_src->back().layout.dtype != m_dst->layout.dtype) {
        return false;
    }

    auto elparam = make_elemwise_op_param<2>();
    auto &src0 = elparam[0], &src1 = elparam[1];
    /* why shape changes?
    std::cout << (*m_src)[0].layout.to_string() << std::endl;
    std::cout << (*m_src)[1].layout.to_string() << std::endl;
    std::cout << src0.layout.to_string() << std::endl;
    std::cout << src1.layout.to_string() << std::endl;
    */
    if (src0.layout.dtype != dtype::Float32{}) {
        return false;
    }

#define DISPATCH_XNNPACK(_mode, _func)                      \
    case Mode::_mode:                                       \
        MEGDNN_DISPATCH_CPU_KERN_OPR(_func(asptr, aslen, bsptr, bslen, aptr, bptr, dptr)); \
        return true


#define DISPATCH_MODE_FLOAT                                 \
    switch (param().mode) {                                                    \
        DISPATCH_XNNPACK(ADD, xnnAdd);\
        DISPATCH_XNNPACK(SUB, xnnSub);\
        DISPATCH_XNNPACK(MUL, xnnMul);\
        DISPATCH_XNNPACK(TRUE_DIV, xnnDiv);\
        DISPATCH_XNNPACK(MIN, xnnMin);\
        DISPATCH_XNNPACK(MAX, xnnMax);\
        default:                                                               \
            break;                                                             \
    }

    auto asptr = (*m_src)[0].layout.shape, bsptr = (*m_src)[1].layout.shape;
    auto aslen = (*m_src)[0].layout.ndim, bslen = (*m_src)[1].layout.ndim;
    auto aptr = (*m_src)[0].ptr<dt_float32>(),
        bptr = (*m_src)[1].ptr<dt_float32>(),
        dptr = m_dst->ptr<dt_float32>();
    DISPATCH_MODE_FLOAT

#undef DISPATCH_MODE_FLOAT
#undef DISPATCH_XNNPACK

    return false;
}

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    if (!dst.layout.is_contiguous())
        return fallback::ElemwiseImpl::exec(srcs, dst);

    m_src = &srcs;
    m_dst = &dst;

    bool optimizing = false;
    optimizing |= m_dst->layout.dtype == dtype::Float32();
    optimizing |= m_dst->layout.dtype == dtype::Int32();

    if (optimizing) {
        if (exec_unary()) {
            return;
        }
        if (exec_binary()) {
            return;
        }
    }

    fallback::ElemwiseImpl::exec(srcs, dst);
}

// vim: syntax=cpp.doxygen