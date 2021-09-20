/**
 * \file dnn/src/wasm/elemwise/binary_base.h
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

void xnn_binary_f32(const size_t* a_shape_ptr, const size_t a_shape_len,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const dt_float32* input_a, const dt_float32* input_b, dt_float32* output, 
                    xnn_create_binary_op create_op, xnn_setup_binary_op setup_op){
    xnn_operator_t binary_op = nullptr;
    const float sum_min = -std::numeric_limits<float>::infinity(),
                sum_max = std::numeric_limits<float>::infinity();
    const uint32_t flags = 0;
    xnn_status status = create_op(sum_min, sum_max, flags, &binary_op);
    if (status != xnn_status_success) {
    megdnn_throw(ssprintf(
        "XNN status for xnn_create_*_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs."));
    }
    status = setup_op(binary_op, a_shape_len, a_shape_ptr, b_shape_len, b_shape_ptr,
                input_a, input_b, output, nullptr);
    if (status != xnn_status_success) {
        megdnn_throw(ssprintf(
            "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
            "status %d. Use -c dbg to see XNN logs.",
            status));
    }

    xnn_run_operator(binary_op, nullptr);
}

typedef xnn_status (*xnn_create_binary_minmax_op)(uint32_t, xnn_operator_t*);
void xnn_binary_f32(const size_t* a_shape_ptr, const size_t a_shape_len,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const dt_float32* input_a, const dt_float32* input_b, dt_float32* output, 
                    xnn_create_binary_minmax_op create_op, xnn_setup_binary_op setup_op){
    xnn_operator_t binary_op = nullptr;
    const uint32_t flags = 0;
    xnn_status status = create_op(flags, &binary_op);
    if (status != xnn_status_success) {
    megdnn_throw(ssprintf(
        "XNN status for xnn_create_*_nd_f32 is not successful. Got "
        "status %d. Use -c dbg to see XNN logs."));
    }
    status = setup_op(binary_op, a_shape_len, a_shape_ptr, b_shape_len, b_shape_ptr,
                input_a, input_b, output, nullptr);
    if (status != xnn_status_success) {
        megdnn_throw(ssprintf(
            "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
            "status %d. Use -c dbg to see XNN logs.",
            status));
    }

    xnn_run_operator(binary_op, nullptr);
}



} // namespace wasm
} // namespace megdnn