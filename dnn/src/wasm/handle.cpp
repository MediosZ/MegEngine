/**
 * \file dnn/src/wasm/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */

#include "src/wasm/handle.h"

#include "src/common/handle_impl.h"

// #include "src/wasm/elemwise/opr_impl.h"
static size_t g_image2d_pitch_alignment = 1;

namespace megdnn {
namespace wasm {

HandleImpl::HandleImpl(megcoreComputingHandle_t computing_handle,
                       HandleType type)
        : HandleImplHelper(computing_handle, type),
          m_dispatcher{megcoreGetCPUDispatcher(computing_handle)} {}

size_t HandleImpl::image2d_pitch_alignment() const {
    return g_image2d_pitch_alignment;
}

HandleImpl::HandleVendorType HandleImpl::vendor_type() const {
    return HandleVendorType::NOT_SPEC;
}

size_t HandleImpl::exchange_image2d_pitch_alignment(size_t alignment) {
    auto ret = g_image2d_pitch_alignment;
    g_image2d_pitch_alignment = alignment;
    return ret;
}
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_SPECIALIZE_CREATE_OPERATOR)

}  // namespace wasm
}  // namespace megdnn
// vim: syntax=cpp.doxygen
