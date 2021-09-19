/**
 * \file dnn/src/wasm/conv_bias/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "src/wasm/conv_bias/algos.h"
#include "src/wasm/conv_bias/conv1x1/algos.h"
#include "src/wasm/conv_bias/conv1x1/algos_conv1x1_gemv.h"
#include "src/wasm/conv_bias/im2col/algos.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/naive/convolution/helper.h"
#include "src/common/algo_base.h"

#include "midout.h"

using namespace megdnn;
using namespace wasm;

namespace {

param::Convolution get_param_convolution(
        const ConvBiasImpl::NCBKernSizeParam& param) {
    param::Convolution::Mode mode;
    param::Convolution::Sparse sparse;
    if (param.filter_meta.should_flip) {
        mode = param::Convolution::Mode::CONVOLUTION;
    } else {
        mode = param::Convolution::Mode::CROSS_CORRELATION;
    }
    return param::Convolution{mode,
                              param.filter_meta.padding[0],
                              param.filter_meta.padding[1],
                              param.filter_meta.stride[0],
                              param.filter_meta.stride[1],
                              param.filter_meta.dilation[1],
                              param.filter_meta.dilation[0],
                              sparse = param::Convolution::Sparse::DENSE,
                              param.filter_meta.format};
}

TensorLayoutArray get_layouts(const ConvBiasImpl::NCBKernSizeParam& p) {
    megdnn_assert(p.filter_meta.format == param::ConvBias::Format::NCHW);
    UNPACK_CONV_NCB_KERN_SIZES(p);
    MEGDNN_MARK_USED_VAR(SH);
    MEGDNN_MARK_USED_VAR(SW);
    MEGDNN_MARK_USED_VAR(PH);
    MEGDNN_MARK_USED_VAR(PW);
    MEGDNN_MARK_USED_VAR(OW);
    MEGDNN_MARK_USED_VAR(OH);
    MEGDNN_MARK_USED_VAR(N);
    TensorLayout src_layout({1, IC, IH, IW}, p.src_type);
    TensorLayout filter_layout({OC, IC, FH, FW}, p.filter_type);
    TensorLayout bias_layout{{}, p.bias_type};
    if (p.bias_mode == BiasMode::BROADCAST_CHANNEL_BIAS) {
        bias_layout = TensorLayout({1, OC, 1, 1}, p.bias_type);
    } else if (p.bias_mode == BiasMode::BIAS) {
        bias_layout = TensorLayout({1, OC, OH, OW}, p.bias_type);
    }
    TensorLayout dst_layout = TensorLayout({1, OC, OH, OW}, p.dst_type);
    return {src_layout, filter_layout, bias_layout, dst_layout};
}

void kern_default(const ConvBiasImpl::NCBKernParam& p) {
    dt_byte* workspace_ptr = static_cast<dt_byte*>(p.workspace_ptr);

    auto filter_meta_ptr =
            reinterpret_cast<const ConvBiasForward::CanonizedFilterMeta*>(
                    &p.filter_meta);
    auto filter_meta = *filter_meta_ptr;
    auto layouts = get_layouts(p);

    TensorND src{reinterpret_cast<dt_byte*>(const_cast<void*>(p.src_ptr)),
                 layouts[0]};
    TensorND filter{const_cast<void*>(p.filter_ptr), layouts[1]};
    auto bias_ptr = reinterpret_cast<dt_byte*>(const_cast<void*>(p.bias_ptr));
    TensorND bias{bias_ptr, layouts[2]};
    TensorND dst{reinterpret_cast<dt_byte*>(const_cast<void*>(p.dst_ptr)),
                 layouts[3]};

    auto sfb = dst;
    if (bias.layout.dtype.enumv() != dst.layout.dtype.enumv()) {
        // intermediate result
        sfb = TensorND{workspace_ptr,
                       TensorLayout{dst.layout, bias.layout.dtype}};
    }
#define DISPATCH_RAW(in_dt, bias_dt, out_dt, cmode, func)                      \
    else if (src.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv &&    \
             filter.layout.dtype.enumv() == DTypeTrait<dtype::in_dt>::enumv && \
             (!bias.layout.dtype.valid() ||                                    \
              bias.layout.dtype.enumv() ==                                     \
                      DTypeTrait<dtype::bias_dt>::enumv) &&                    \
             sfb.layout.dtype.enumv() == DTypeTrait<dtype::out_dt>::enumv &&   \
             p.compute_mode == param::ConvBias::ComputeMode::cmode) {          \
        func(src, filter, bias, sfb, workspace_ptr, filter_meta);              \
    }
#define DISPATCH(in_dt, out_dt)                             \
    DISPATCH_RAW(in_dt, out_dt, out_dt, DEFAULT,            \
                 (megdnn::naive::convolution::forward_bias< \
                         DTypeTrait<dtype::in_dt>::ctype,   \
                         DTypeTrait<dtype::in_dt>::ctype,   \
                         DTypeTrait<dtype::out_dt>::ctype,  \
                         DTypeTrait<dtype::out_dt>::ctype>))
    if (0) {
    }
    DISPATCH(Float32, Float32)
    DISPATCH(Int8, Int16)
    DISPATCH(Int8, Int32)
    DISPATCH(QuantizedS8, QuantizedS32)
    DISPATCH(Quantized8Asymm, QuantizedS32)
#if !MEGDNN_DISABLE_FLOAT16
    DISPATCH(Float16, Float16)
    DISPATCH_RAW(
            Float16, Float16, Float16, FLOAT32,
            (megdnn::naive::convolution::forward_bias<dt_float16, dt_float16,
                                                      dt_float16, dt_float32>))
#endif
    else {
        megdnn_throw(
                ssprintf("unsupported naive ConvBias(%s, %s, %s) -> %s",
                         src.layout.dtype.name(), filter.layout.dtype.name(),
                         bias.layout.dtype.name(), dst.layout.dtype.name()));
    }
#undef DISPATCH
#undef DISPATCH_RAW

    auto res = sfb;
    using NonlineMode = param::ConvBias::NonlineMode;
    switch (p.nonlineMode) {
#define cb(_mode)                                                             \
    case NonlineMode::_mode: {                                                \
        if (res.layout.dtype.category() != DTypeCategory::QUANTIZED) {        \
            auto nonlinear =                                                  \
                    inplace_cpu_handle()->create_operator<ElemwiseForward>(); \
            nonlinear->param().mode = Elemwise::Param::Mode::_mode;           \
            nonlinear->exec({res}, dst);                                      \
        } else {                                                              \
            auto nonlinear = inplace_cpu_handle()                             \
                                     ->create_operator<ElemwiseMultiType>();  \
            nonlinear->param().mode =                                         \
                    ElemwiseMultiType::Param::Mode::Q##_mode;                 \
            nonlinear->exec({res}, dst);                                      \
        }                                                                     \
        break;                                                                \
    }
        cb(RELU);
        cb(H_SWISH);
#undef cb
        case NonlineMode::SIGMOID: {
            megdnn_assert(res.layout.dtype.category() !=
                          DTypeCategory::QUANTIZED);
            auto nonlinear =
                    inplace_cpu_handle()->create_operator<ElemwiseForward>();
            nonlinear->param().mode = Elemwise::Param::Mode::SIGMOID;
            nonlinear->exec({res}, res);
            if (res.raw_ptr != dst.raw_ptr) {
                inplace_cpu_handle()->create_operator<TypeCvt>()->exec(res,
                                                                       dst);
            }
            break;
        }
        case NonlineMode::IDENTITY: {
            if (res.raw_ptr != dst.raw_ptr) {
                inplace_cpu_handle()->create_operator<TypeCvt>()->exec(res,
                                                                       dst);
            }
            break;
        }
        default:
            megdnn_assert(false);
    }
}
}  // namespace

MIDOUT_DECL(megdnn_wasm_naive)

/* ======================= AlgoNaive ======================== */

bool ConvBiasImpl::AlgoNaive::usable(
        const NCBKernSizeParam& param,
        AlgoSelectionStrategy /*algo_selection_strategy*/) const {
    MIDOUT_BEGIN(megdnn_wasm_naive, 0) {
        auto algo_data_type = param.deduce_algo_data_type();
        return param.filter_meta.format == param::ConvBias::Format::NCHW &&
               contain_data_type(get_algo_type().data_type, algo_data_type);
    }
    MIDOUT_END();
    return false;
}

size_t ConvBiasImpl::AlgoNaive::get_workspace(const NCBKernSizeParam& p) const {
    MIDOUT_BEGIN(megdnn_wasm_naive, 1) {
        auto layouts = get_layouts(p);
        //! When group>1 or n>1, this algo will parallel by group and n
        size_t nr_threads = p.nr_threads;
        auto conv_opr =
                inplace_cpu_handle()->create_operator<ConvolutionForward>();
        conv_opr->param() = get_param_convolution(p);
        if (p.dst_type.enumv() == DTypeEnum::QuantizedS8 ||
            p.dst_type.enumv() == DTypeEnum::Quantized8Asymm) {
            TensorLayout conv_dst_layout;
            conv_opr->deduce_layout(layouts[0], layouts[1], conv_dst_layout);
            WorkspaceBundle bundle(nullptr,
                                   {conv_dst_layout.span().dist_byte()});
            return bundle.total_size_in_bytes() * nr_threads;
        }
        return 0;
    }
    MIDOUT_END();
    return 0;
}

SmallVector<ConvBiasImpl::NCBKern> ConvBiasImpl::AlgoNaive::dispatch_kerns(
        const NCBKernSizeParam& p) const {
    size_t workspace_size = get_workspace(p);
    //! When group>1 or n>1, this algo will parallel by group and n
    size_t nr_threads = p.nr_threads;
    size_t GROUP = p.filter_meta.group;
    size_t N = p.n;
    size_t workspace_per_thread = workspace_size / nr_threads;
    auto kern = [workspace_per_thread](
                        const NCBKernParam& param,
                        const NCBKernIndex& ncb_index) {
        MIDOUT_BEGIN(megdnn_wasm_naive, 2) {
            size_t group_id = ncb_index.ndrange_id[0];
            size_t batch_id = ncb_index.ndrange_id[1];
            size_t thread_id = ncb_index.thread_id;
            auto thread_param = param;
            thread_param.workspace_ptr = reinterpret_cast<void*>(
                    reinterpret_cast<ptrdiff_t>(param.workspace_ptr) +
                    thread_id * workspace_per_thread);
            thread_param.filter_ptr = param.filter<void>(group_id);
            thread_param.dst_ptr = param.dst<void>(batch_id, group_id);
            thread_param.src_ptr = param.src<void>(batch_id, group_id);
            thread_param.bias_ptr = param.bias<void>(batch_id, group_id);
            kern_default(thread_param);
        }
        MIDOUT_END();
    };
    return {{kern, {GROUP, N, 1_z}}};
}

// vim: syntax=cpp.doxygen
