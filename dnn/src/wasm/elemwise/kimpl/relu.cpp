/**
 * \file dnn/src/wasm/elemwise/kimpl/relu.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./unary_base.h"
namespace megdnn {
namespace wasm {

void xnnRelu(const dt_float32* input, dt_float32* output, int32_t n) {
    const float min = 0;
    const float max = std::numeric_limits<float>::infinity();
    xnn_clamp_f32(input, output, n, min, max);
}

} // namespace wasm
} // namespace megdnn