/**
 * \file dnn/src/wasm/elemwise/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/elemwise/opr_impl.h"
#include "src/wasm/elemwise/kimpl/unary.h"
#include "src/wasm/elemwise/kimpl/binary.h"

namespace megdnn {
namespace wasm {

class ElemwiseImpl final: public fallback::ElemwiseImpl {
    bool exec_unary();
    bool exec_binary();
    bool exec_ternary_fma3();

    public:
        using fallback::ElemwiseImpl::ElemwiseImpl;
        void exec(const TensorNDArray &srcs,
                _megdnn_tensor_out dst) override;
};

} // namespace wasm
} // namespace megdnn

// vim: syntax=cpp.doxygen

