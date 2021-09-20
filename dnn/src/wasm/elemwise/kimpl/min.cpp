/**
 * \file dnn/src/wasm/elemwise/kimpl/min.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "./binary_base.h"
namespace megdnn {
namespace wasm {

void xnnMin(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output) {
    xnn_binary_f32(a_shape_ptr, a_shape_len, 
                    b_shape_ptr, b_shape_len, 
                    input_a, input_b, output, 
                    xnn_create_minimum_nd_f32, xnn_setup_minimum_nd_f32);
}

} // namespace wasm
} // namespace megdnn