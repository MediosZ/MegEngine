/**
 * \file dnn/src/x86/elemwise/opr_impl.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/x86/elemwise/opr_impl.h"
#include "src/x86/elemwise_op.h"
#include "src/x86/utils.h"

#include "src/common/utils.h"
#include "src/naive/handle.h"
#include <iostream>

using namespace megdnn;
using namespace x86;

void ElemwiseImpl::exec(const TensorNDArray& srcs, _megdnn_tensor_out dst) {
    if (!dst.layout.is_contiguous())
        return fallback::ElemwiseImpl::exec(srcs, dst);

    m_src = &srcs;
    m_dst = &dst;

    bool optimizing = false;
    optimizing |= m_dst->layout.dtype == dtype::Float32();
    optimizing |= m_dst->layout.dtype == dtype::Int32();
    
    std::cout << "elemwise exec with optmizing "<< (optimizing ? "true" : "false") << std::endl;

    fallback::ElemwiseImpl::exec(srcs, dst);
}

// vim: syntax=cpp.doxygen
