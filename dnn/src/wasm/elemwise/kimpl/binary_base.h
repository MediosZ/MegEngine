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
#pragma once

#include <xnnpack.h>
#include "src/common/utils.h"

namespace megdnn {
namespace wasm {
typedef xnn_status (*xnn_create_binary_op)(float, float, uint32_t,
                                           xnn_operator_t*);
typedef xnn_status (*xnn_setup_binary_op)(xnn_operator_t, size_t, const size_t*,
                                          size_t, const size_t*, const float*,
                                          const float*, float*, pthreadpool_t);

void xnn_binary_f32(const size_t* a_shape_ptr, const size_t a_shape_len,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const dt_float32* input_a, const dt_float32* input_b, dt_float32* output, 
                    xnn_create_binary_op create_op, xnn_setup_binary_op setup_op);

typedef xnn_status (*xnn_create_binary_minmax_op)(uint32_t, xnn_operator_t*);
void xnn_binary_f32(const size_t* a_shape_ptr, const size_t a_shape_len,
                    const size_t* b_shape_ptr, const size_t b_shape_len,
                    const dt_float32* input_a, const dt_float32* input_b, dt_float32* output, 
                    xnn_create_binary_minmax_op create_op, xnn_setup_binary_op setup_op);

} // namespace wasm
} // namespace megdnn