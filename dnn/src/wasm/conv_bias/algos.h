/**
 * \file dnn/src/wasm/conv_bias/algos.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "src/wasm/conv_bias/opr_impl.h"
#include "src/wasm/matrix_mul/opr_impl.h"
#include "megdnn/thin/small_vector.h"

namespace megdnn {
namespace wasm {

class ConvBiasImpl::AlgoNaive final : public AlgoBase {
public:
    AlgoAttribute attribute() const override{
        return AlgoAttribute::REPRODUCIBLE | AlgoAttribute::NAIVE;
    }
    const char* name() const override { return "WASM_NAIVE"; }
    bool usable(const NCBKernSizeParam& param,
                AlgoSelectionStrategy algo_selection_strategy) const override;
    size_t get_workspace(const NCBKernSizeParam& param) const override;
    SmallVector<NCBKern> dispatch_kerns(const NCBKernSizeParam&) const override;

    ConvAlgoTypePack get_algo_type() const override {
        auto support_data_type = static_cast<AlgoDataType>(
                static_cast<uint32_t>(AlgoDataType::FLOAT16) |
                static_cast<uint32_t>(AlgoDataType::FLOAT32) |
                static_cast<uint32_t>(AlgoDataType::INT8X8X16) |
                static_cast<uint32_t>(AlgoDataType::QINT8X8X32) |
                static_cast<uint32_t>(AlgoDataType::QUINT8X8X32));
        return {support_data_type, AlgoCategory::NAIVE};
    }
    MEGDNN_DECL_ALGO_TYPE(WASM_NAIVE)
};

}  // namespace wasm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
