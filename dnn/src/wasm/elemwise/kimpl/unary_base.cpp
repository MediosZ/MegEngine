/**
 * \file dnn/src/wasm/elemwise/unary_base.h
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

void xnn_unary_f32(const dt_float32* input, dt_float32* output, int32_t n, 
                    xnn_create_unary_op create_op, xnn_setup_unary_op setup_op) {
    xnn_operator_t unary_op = nullptr;
    const size_t channels = 1, input_stride = 1, output_stride = 1;
    const uint32_t flags = 1;

    xnn_status status = create_op(channels, input_stride, output_stride, flags, &unary_op);
    if (status != xnn_status_success) {
      megdnn_throw(ssprintf(
          "XNN status for xnn_create_*_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs."));
    }

    status = setup_op(unary_op, n, input, output, nullptr);
    if (status != xnn_status_success) {
        megdnn_throw(ssprintf(
            "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
            "status %d. Use -c dbg to see XNN logs.",
            status));
    }

    xnn_run_operator(unary_op, nullptr);
}


void xnn_clamp_f32(const dt_float32* input, dt_float32* output, int32_t n, const float min, const float max){
    xnn_operator_t unary_op = nullptr;
    const size_t channels = 1, input_stride = 1, output_stride = 1;
    const uint32_t flags = 1;

    xnn_status status = xnn_create_clamp_nc_f32(channels, input_stride, output_stride, min, max, flags, &unary_op);
    if (status != xnn_status_success) {
      megdnn_throw(ssprintf(
          "XNN status for xnn_create_*_nd_f32 is not successful. Got "
          "status %d. Use -c dbg to see XNN logs."));
    }

    status = xnn_setup_clamp_nc_f32(unary_op, n, input, output, nullptr);
    if (status != xnn_status_success) {
        megdnn_throw(ssprintf(
            "XNN status for xnn_setup_*_nd_f32 is not successful. Got "
            "status %d. Use -c dbg to see XNN logs.",
            status));
    }

    xnn_run_operator(unary_op, nullptr);
}



} // namespace wasm
} // namespace megdnn