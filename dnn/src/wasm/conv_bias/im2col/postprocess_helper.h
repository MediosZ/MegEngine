/**
 * \file dnn/src/wasm/conv_bias/postprocess_helper.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include "megdnn/basic_types.h"
// #include "src/wasm/conv_bias/opr_impl.h"

#include "midout.h"

MIDOUT_DECL(wasm_conv_bias_postprocess_helper)

namespace {

template <typename ctype, typename dtype = ctype,
          megdnn::PostprocessMode postprocess_mode =
                  megdnn::PostprocessMode::FLOAT>
struct PostProcess {
    static void run(void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        // FOR_BIAS(bias_mode)
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::NO_PROCESS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        MEGDNN_MARK_USED_VAR(conv_dst_ptr);
        MEGDNN_MARK_USED_VAR(bias_ptr);
        MEGDNN_MARK_USED_VAR(dst_ptr);
        MEGDNN_MARK_USED_VAR(bias_mode);
        MEGDNN_MARK_USED_VAR(nonlineMode);
        MEGDNN_MARK_USED_VAR(bias_type);
        MEGDNN_MARK_USED_VAR(dst_type);
        MEGDNN_MARK_USED_VAR(N);
        MEGDNN_MARK_USED_VAR(OC);
        MEGDNN_MARK_USED_VAR(OH);
        MEGDNN_MARK_USED_VAR(OW);
        MEGDNN_MARK_USED_VAR(pack_oc_size);
        megdnn_throw_if(bias_mode != megdnn::BiasMode::NO_BIAS ||
                                nonlineMode != megdnn::NonlineMode::IDENTITY,
                        megdnn_error, "biasmode or nonlineMode do not support");
    }
};

template <typename opctype, typename opdtype>
struct PostProcess<opctype, opdtype, megdnn::PostprocessMode::QUANTIZED> {
    static void run(void* conv_dst_ptr, const void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        //! when OH * OW = 1, the bias_mode will be BiasMode::BIAS. It is wrong,
        //! we deal this case at default branch.
        // FOR_BIAS(bias_mode, OH, OW);
    }
};

template <typename ctype, typename dtype>
struct PostProcess<ctype, dtype, megdnn::PostprocessMode::ADD_BIAS> {
    static void run(void* conv_dst_ptr, void* bias_ptr, void* dst_ptr,
                    megdnn::BiasMode bias_mode, megdnn::NonlineMode nonlineMode,
                    megdnn::DType bias_type, megdnn::DType dst_type, size_t N,
                    size_t OC, size_t OH, size_t OW, size_t pack_oc_size = 1) {
        megdnn_throw_if(nonlineMode != megdnn::NonlineMode::IDENTITY,
                        megdnn_error, "nonlineMode do not support");
        // FOR_BIAS(bias_mode, OH, OW);
    }
};

}  // namespace
