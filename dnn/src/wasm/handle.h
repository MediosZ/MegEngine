/**
 * \file dnn/src/wasm/handle.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#pragma once
#include "src/fallback/handle.h"
#include <xnnpack.h>
namespace megdnn {
namespace wasm {

class HandleImpl : public fallback::HandleImpl {
public:
    HandleImpl(megcoreComputingHandle_t computing_handle,
               HandleType type = HandleType::WASM);

    template <typename Opr>
    std::unique_ptr<Opr> create_operator();

    size_t alignment_requirement() const override;

};

}  // namespace wasm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
