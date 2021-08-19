/**
 * \file dnn/src/wasm/elemwise/unary.h
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

void xnnSigmoid(const dt_float32* input, dt_float32* output, int32_t n);
void xnnRound(const dt_float32* input, dt_float32* output, int32_t n);
void xnnRelu(const dt_float32* input, dt_float32* output, int32_t n);
void xnnAbs(const dt_float32* input, dt_float32* output, int32_t n);
void xnnCeil(const dt_float32* input, dt_float32* output, int32_t n);
void xnnFloor(const dt_float32* input, dt_float32* output, int32_t n);
} // namespace wasm
} // namespace megdnn

