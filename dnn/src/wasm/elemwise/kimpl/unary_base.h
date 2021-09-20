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
#pragma once

#include <xnnpack.h>
#include "src/common/utils.h"

namespace megdnn {
namespace wasm {
typedef xnn_status (*xnn_create_unary_op)(size_t, size_t, size_t, uint32_t,
                                        xnn_operator_t*);
typedef xnn_status (*xnn_setup_unary_op)(xnn_operator_t, size_t, const float*,
                                        float*, pthreadpool_t);


void xnn_unary_f32(const dt_float32* input, dt_float32* output, int32_t n, 
                    xnn_create_unary_op create_op, xnn_setup_unary_op setup_op);

void xnn_clamp_f32(const dt_float32* input, dt_float32* output, int32_t n, const float min, const float max);


} // namespace wasm
} // namespace megdnn