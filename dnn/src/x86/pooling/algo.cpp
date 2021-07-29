/**
 * \file dnn/src/x86/pooling/algos.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#include "src/x86/pooling/algo.h"
#include "megdnn/opr_param_defs.h"
#include "src/common/opr_delegate.h"
#include "src/common/utils.h"
#include "src/fallback/pooling/opr_impl.h"
#include "src/naive/handle.h"
#include "src/x86/handle.h"
#include "src/x86/pooling/do_max_pooling_3x3_s2x2_float_sse.h"
#include "src/x86/pooling/pooling_special_cases.h"
#include "src/x86/utils.h"

using namespace megdnn;
using namespace x86;

namespace {

#if MEGDNN_X86_WITH_MKL_DNN
template <dnnl::memory::format_tag format_tag, bool use_mkl_mem>
dnnl::memory tensor_to_mkl_memory(_megdnn_tensor_in src,
                                  const dnnl::engine& mkldnn_eng,
                                  dnnl::memory::data_type mkldnn_datatype) {
    megdnn_assert(format_tag == dnnl::memory::format_tag::nChw8c ||
                          format_tag == dnnl::memory::format_tag::nchw ||
                          format_tag == dnnl::memory::format_tag::nhwc,
                  "not support format");

    dnnl::memory::dims src_shape = {
            static_cast<long>(src.layout[0]), static_cast<long>(src.layout[1]),
            static_cast<long>(src.layout[2]), static_cast<long>(src.layout[3])};
    if (format_tag == dnnl::memory::format_tag::nChw8c) {
        src_shape = {static_cast<long>(src.layout[0]),
                     static_cast<long>(src.layout[1] * 8),
                     static_cast<long>(src.layout[2]),
                     static_cast<long>(src.layout[3])};
    }
    auto megdnn_src_md =
            dnnl::memory::desc({src_shape}, mkldnn_datatype, format_tag);
    if (use_mkl_mem) {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng);
        return megdnn_src_memory;
    } else {
        auto megdnn_src_memory = dnnl::memory(megdnn_src_md, mkldnn_eng,
                                              const_cast<void*>(src.raw_ptr));
        return megdnn_src_memory;
    }
}

#endif

}  // namespace

PoolingImpl::AlgoPack::AlgoPack() {
    all_algos.push_back(&algo_mean_w2s2_avx);
    all_algos.push_back(&algo_mean_w2s2_sse3);
    all_algos.push_back(&algo_max_w2s2_sse);
    all_algos.push_back(&algo_max_w3s3_sse);
#if MEGDNN_X86_WITH_MKL_DNN
    all_algos.push_back(&algo_mkldnn_nchw);
    all_algos.push_back(&algo_mkldnn_nchw88);
#endif
    all_algos.push_back(&algo_fallback);

    for (auto&& algo : all_algos) {
        m_all_algos_map.emplace(algo->info().desc, algo);
    }
}

PoolingImpl::AlgoPack PoolingImpl::sm_algo_pack;
MEGDNN_DEF_GET_ALGO_FROM_DESC(PoolingImpl)

PoolingImpl::AlgoBase::SizeArgs::SizeArgs(PoolingImpl* o,
                                          const TensorLayout& src,
                                          const TensorLayout& dst)
        : handle{static_cast<x86::HandleImpl*>(o->handle())},
          opr{o},
          layout_src{src},
          layout_dst{dst} {}

PoolingImpl::AlgoBase::ExecArgs::ExecArgs(PoolingImpl* opr,
                                          _megdnn_tensor_in src,
                                          _megdnn_tensor_out dst,
                                          _megdnn_workspace workspace)
        : SizeArgs(opr, src.layout, dst.layout),
          src_tensor{&src},
          dst_tensor{&dst},
          workspace{workspace} {}

std::string PoolingImpl::AlgoBase::SizeArgs::to_string() const {
    return ssprintf("src=%s, dst=%s", layout_src.to_string().c_str(),
                    layout_dst.to_string().c_str());
}

bool PoolingImpl::AlgoMeanW2S2AVX::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::AVX) &&
            args.opr->param().mode == Mode::AVERAGE &&
            args.opr->param().format == Param::Format::NCHW &&
            args.layout_src.dtype == dtype::Float32() && FH == 2 && FW == 2 &&
            SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMeanW2S2AVX::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor->raw_ptr);
    auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor->raw_ptr);
    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
        mean_pooling_w2x2_s2x2_avx(sptr + n * C * IH * IW + c * IH * IW, IH, IW,
                                   dptr + n * C * OH * OW + c * OH * OW, OH, OW,
                                   PH, PW, true);
    });
}

bool PoolingImpl::AlgoMeanW2S2SSE3::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE3) &&
            args.opr->param().mode == Mode::AVERAGE &&
            args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().format == Param::Format::NCHW && FH == 2 &&
            FW == 2 && SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMeanW2S2SSE3::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor->raw_ptr);
    auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor->raw_ptr);
    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
        mean_pooling_w2x2_s2x2_sse3(sptr + n * C * IH * IW + c * IH * IW, IH,
                                    IW, dptr + n * C * OH * OW + c * OH * OW,
                                    OH, OW, PH, PW, true);
    });
}

bool PoolingImpl::AlgoMaxW2S2SSE::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE) &&
            args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW && FH == 2 &&
            FW == 2 && SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMaxW2S2SSE::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor->raw_ptr);
    auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor->raw_ptr);
    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(rep(n, N) rep(c, C) {
        max_pooling_w2x2_s2x2_sse(sptr + n * C * IH * IW + c * IH * IW, IH, IW,
                                  dptr + n * C * OH * OW + c * OH * OW, OH, OW,
                                  PH, PW);
    });
}

bool PoolingImpl::AlgoMaxW3S3SSE::is_available(const SizeArgs& args) const {
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;

    return (is_supported(SIMDType::SSE) &&
            args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW && FH == 3 &&
            FW == 3 && SH == 2 && SW == 2);
}

void PoolingImpl::AlgoMaxW3S3SSE::exec(const ExecArgs& args) const {
    auto N = args.layout_src.shape[0];
    auto C = args.layout_src.shape[1];
    auto IH = args.layout_src.shape[2];
    auto IW = args.layout_src.shape[3];
    auto OH = args.layout_dst.shape[2];
    auto OW = args.layout_dst.shape[3];
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto sptr = reinterpret_cast<dt_float32*>(args.src_tensor->raw_ptr);
    auto dptr = reinterpret_cast<dt_float32*>(args.dst_tensor->raw_ptr);
    auto handle = [=]() { return args.handle; };
    MEGDNN_DISPATCH_CPU_KERN_OPR(
            WorkspaceBundle ws = get_bundle(args.layout_src, args.layout_dst,
                                            args.opr->param());
            ws.set(args.workspace.raw_ptr); rep(n, N) rep(c, C) {
                do_max_pooling_3x3_s2x2_float_SSE(
                        sptr + n * C * IH * IW + c * IH * IW,
                        dptr + n * C * OH * OW + c * OH * OW, IH, IW, OH, OW,
                        PH, PW, ws);
            });
}

#if MEGDNN_X86_WITH_MKL_DNN
bool PoolingImpl::AlgoMKLDNNNCHW::is_available(const SizeArgs& args) const {
    return ((args.layout_src.dtype.enumv() == DTypeEnum::QuantizedS8 ||
             args.layout_src.dtype.enumv() == DTypeEnum::Int8) &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW);
}

void PoolingImpl::AlgoMKLDNNNCHW::exec(const ExecArgs& args) const {
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto handle = [=]() { return args.handle; };

    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    auto mkldnn_eng = x86_handle->mkldnn_engine();
    auto mkldnn_stream = x86_handle->mkldnn_stream();
    auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
    dnnl::memory::dims pool_strides = {SH, SW};
    dnnl::memory::dims pool_padding = {PH, PW};
    dnnl::memory::dims pool_kernel = {FH, FW};

    dnnl::memory&& megdnn_src_memory_ori =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                    *args.src_tensor, mkldnn_eng, dnnl::memory::data_type::s8);
    dnnl::memory&& megdnn_dst_memory_ori =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nchw, false>(
                    *args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::s8);

    dnnl::memory&& megdnn_src_memory =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                    *args.src_tensor, mkldnn_eng, dnnl::memory::data_type::s8);
    dnnl::memory&& megdnn_dst_memory =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nhwc, true>(
                    *args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::s8);

    auto reorder_src = dnnl::reorder(megdnn_src_memory_ori, megdnn_src_memory);
    auto reorder_dst = dnnl::reorder(megdnn_dst_memory, megdnn_dst_memory_ori);
    auto pool1_desc = dnnl::pooling_forward::desc(
            dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
            megdnn_src_memory.get_desc(), megdnn_dst_memory.get_desc(),
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd =
            dnnl::pooling_forward::primitive_desc(pool1_desc, mkldnn_eng);
    auto pool = dnnl::pooling_forward(pool_pd);

    auto run = [mkldnn_stream, mkldnn_eng, reorder_src, pool, reorder_dst,
                megdnn_src_memory_ori, megdnn_src_memory, megdnn_dst_memory,
                megdnn_dst_memory_ori](void) {
        MEGDNN_MARK_USED_VAR(mkldnn_eng);
        auto mkl_stream = mkldnn_stream;
        reorder_src.execute(mkl_stream, {{DNNL_ARG_FROM, megdnn_src_memory_ori},
                                         {DNNL_ARG_TO, megdnn_src_memory}});
        pool.execute(mkl_stream, {{DNNL_ARG_SRC, megdnn_src_memory},
                                  {DNNL_ARG_DST, megdnn_dst_memory}});
        reorder_dst.execute(mkl_stream, {{DNNL_ARG_FROM, megdnn_dst_memory},
                                         {DNNL_ARG_TO, megdnn_dst_memory_ori}});
        mkl_stream.wait();
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(run());
}

#endif

#if MEGDNN_X86_WITH_MKL_DNN
bool PoolingImpl::AlgoMKLDNNNCHW88::is_available(const SizeArgs& args) const {
    return (args.layout_src.dtype == dtype::Float32() &&
            args.opr->param().mode == Mode::MAX &&
            args.opr->param().format == Param::Format::NCHW88);
}

void PoolingImpl::AlgoMKLDNNNCHW88::exec(const ExecArgs& args) const {
    auto PH = args.opr->param().pad_h;
    auto PW = args.opr->param().pad_w;
    auto FH = args.opr->param().window_h;
    auto FW = args.opr->param().window_w;
    auto SH = args.opr->param().stride_h;
    auto SW = args.opr->param().stride_w;
    auto handle = [=]() { return args.handle; };

    auto x86_handle = static_cast<HandleImpl*>(inplace_cpu_handle().get());
    auto mkldnn_eng = x86_handle->mkldnn_engine();
    auto mkldnn_stream = x86_handle->mkldnn_stream();
    auto mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
    switch (args.opr->param().mode) {
        case Mode::MAX:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_max;
            break;
        case Mode::AVERAGE:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_avg_include_padding;
            break;
        case Mode::AVERAGE_COUNT_EXCLUDE_PADDING:
            mkldnn_pooling_mode = dnnl::algorithm::pooling_avg_exclude_padding;
            break;
        default:
            megdnn_throw("not supported pooling mode\n");
    };

    dnnl::memory::dims pool_strides = {SH, SW};
    dnnl::memory::dims pool_padding = {PH, PW};
    dnnl::memory::dims pool_kernel = {FH, FW};
    dnnl::memory&& megdnn_src_memory_ori =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                    *args.src_tensor, mkldnn_eng, dnnl::memory::data_type::f32);
    dnnl::memory&& megdnn_dst_memory_ori =
            tensor_to_mkl_memory<dnnl::memory::format_tag::nChw8c, false>(
                    *args.dst_tensor, mkldnn_eng, dnnl::memory::data_type::f32);
    auto pool_desc = dnnl::pooling_forward::desc(
            dnnl::prop_kind::forward_inference, mkldnn_pooling_mode,
            megdnn_src_memory_ori.get_desc(), megdnn_dst_memory_ori.get_desc(),
            pool_strides, pool_kernel, pool_padding, pool_padding);
    auto pool_pd = dnnl::pooling_forward::primitive_desc(pool_desc, mkldnn_eng);
    auto pool = dnnl::pooling_forward(pool_pd);

    auto run = [mkldnn_stream, pool, mkldnn_eng, megdnn_src_memory_ori,
                megdnn_dst_memory_ori](void) {
        MEGDNN_MARK_USED_VAR(mkldnn_eng);
        auto mkl_stream = mkldnn_stream;

        pool.execute(mkl_stream, {{DNNL_ARG_SRC, megdnn_src_memory_ori},
                                  {DNNL_ARG_DST, megdnn_dst_memory_ori}});
        mkl_stream.wait();
    };
    MEGDNN_DISPATCH_CPU_KERN_OPR(run());
}

#endif