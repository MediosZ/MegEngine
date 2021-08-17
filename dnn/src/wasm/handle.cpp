/**
 * \file dnn/src/wasm/handle.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/common/handle_impl.h"
#include "src/common/version_symbol.h"
#include "src/wasm/handle.h"
#include <iostream>

namespace megdnn {
namespace wasm {

template <typename Opr>
std::unique_ptr<Opr> HandleImpl::create_operator() {
    return fallback::HandleImpl::create_operator<Opr>();
}

HandleImpl::HandleImpl(megcoreComputingHandle_t computing_handle,
                       HandleType type)
        : fallback::HandleImpl::HandleImpl(computing_handle, type) {
    std::cout << "HandleImpl::HandleImpl" << std::endl;
}

size_t HandleImpl::alignment_requirement() const {
    return 32;
}

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpragmas"
#pragma GCC diagnostic ignored "-Winstantiation-after-specialization"
MEGDNN_FOREACH_OPR_CLASS(MEGDNN_INST_CREATE_OPERATOR)
#pragma GCC diagnostic pop

}  // namespace wasm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
