/**
 * \file dnn/src/wasm/elemwise/binary.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/common/utils.h"

namespace megdnn {
namespace wasm {
void xnnAdd(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

void xnnDiv(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

void xnnMax(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

void xnnMin(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

void xnnMul(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

void xnnSub(const size_t* a_shape_ptr, const size_t a_shape_len,
            const size_t* b_shape_ptr, const size_t b_shape_len,
            const dt_float32* input_a, const dt_float32* input_b, dt_float32* output);

} // namespace wasm
} // namespace megdnn