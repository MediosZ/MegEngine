/**
 * \file dnn/src/wasm/matrix_mul/opr_impl.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or
 * implied.
 */
#pragma once
#include <unordered_map>
#include "megdnn/opr_param_defs.h"
#include "megdnn/oprs/base.h"
#include "src/common/algo_base.h"
#include "src/common/utils.h"
#include "src/naive/matrix_mul/opr_impl.h"
#include <xnnpack.h>
namespace megdnn {

struct AlgoTypePack {
    detail::AlgoDataType data_type : 32;
    param::MatrixMul::Format format : 32;
};

namespace wasm {
class MatrixMulImpl : public naive::MatrixMulForwardImpl {
public:
    using naive::MatrixMulForwardImpl::MatrixMulForwardImpl;
    using AlgoDataType = detail::AlgoDataType;

    bool is_thread_safe() const override { return true; }

    size_t get_workspace_in_bytes(const TensorLayout&, const TensorLayout&,
                                  const TensorLayout&) override;

    void exec(_megdnn_tensor_in A, _megdnn_tensor_in B, _megdnn_tensor_out C,
              _megdnn_workspace workspace) override;

    struct KernSizeParam {
        DType A_type, B_type, C_type;
        size_t M, N, K;
        size_t LDA, LDB, LDC;
        bool trA, trB;
        Param::ComputeMode compute_mode;
        Param::Format format;
        //! get the data type category of the param for select the algo
        AlgoDataType deduce_algo_data_type() const;
    };

    struct KernParam : public KernSizeParam {
        const void* A_ptr;
        const void* B_ptr;
        void* C_ptr;
        void* workspace_ptr;
        size_t workspace_size;

        template <typename T>
        inline const T* A() const {
            // A_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(A_ptr);
        }

        template <typename T>
        inline const T* B() const {
            // B_type.assert_is_compatible_ctype<T>();
            return static_cast<const T*>(B_ptr);
        }

        template <typename T>
        inline T* C() const {
            // C_type.assert_is_compatible_ctype<T>();
            return static_cast<T*>(C_ptr);
        }
        template <typename T>
        inline T* workspace() const {
            return static_cast<T*>(workspace_ptr);
        }
    };

    typedef void (*kern_t)(const KernParam&);
    typedef void (*kern_naked_t)(const KernParam&, const void* a_panel,
                                 const void* b_panel);
    class AlgoBase : public Algorithm {
    protected:
        virtual ~AlgoBase() = default;

        bool can_be_treated_as_int8x8x32(const KernSizeParam& param) const {
            return param.A_type.enumv() == param.B_type.enumv() &&
                   (param.A_type.enumv() == DTypeEnum::Int8 ||
                    param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
                   (param.C_type.enumv() == DTypeEnum::Int32 ||
                    param.C_type.enumv() == DTypeEnum::QuantizedS32) &&
                   param.compute_mode == Param::ComputeMode::DEFAULT &&
                   param.format == param::MatrixMul::Format::DEFAULT;
        }

        bool can_be_treated_as_int8x8x16(const KernSizeParam& param) const {
            return param.A_type.enumv() == param.B_type.enumv() &&
                   (param.A_type.enumv() == DTypeEnum::Int8 ||
                    param.A_type.enumv() == DTypeEnum::QuantizedS8) &&
                   (param.C_type.enumv() == DTypeEnum::Int16 ||
                    param.C_type.enumv() == DTypeEnum::QuantizedS16);
        }

    public:
        AlgoBase() { m_handle_type = Handle::HandleType::WASM; }
        enum class AlgoType : uint32_t {
            //! wasm
            WASM_F32K8x12x1 = 1 << 0,
            WASM_GEMV,
            WASM_NAIVE,
        };

        enum class AlgoSet : uint32_t {
            ALGO_TYPE_GEMM = 0,
            ALGO_TYPE_GEMV = 1,
        };

        enum class PackMode : uint32_t {
            DEFAULT = 0,
            NO_PACK = 1,
            ONLY_PACKA = 2,
        };

        struct InnerBlockSize {
            size_t m, n, k;
        };

        struct MatmulDescription {
            PackMode packmode;
            InnerBlockSize innerblocksize;
            AlgoTypePack algo_type;
            size_t packa_type_size;
        };

        virtual bool usable(const KernSizeParam&) const = 0;
        virtual bool preferred(const KernSizeParam&) const { return true; }
        virtual size_t get_workspace(const KernSizeParam&) const = 0;
        virtual kern_t get_kern(const KernSizeParam&) const = 0;
        virtual kern_naked_t get_kern_naked(const KernSizeParam&) const {
            megdnn_assert(0);
        };
        virtual AlgoSet algoset() const { return AlgoSet::ALGO_TYPE_GEMM; }
        virtual PackMode packmode() const { return PackMode::DEFAULT; }
        virtual void pack_A(const KernParam&, void*, size_t, size_t) const {
            megdnn_assert(0);
        };
        virtual void pack_B(const KernParam&, void*, size_t, size_t) const {
            megdnn_assert(0);
        };
        virtual WorkspaceBundle get_bundle(const KernSizeParam&) const {
            megdnn_assert(0);
        };
        virtual InnerBlockSize get_inner_block_size() const {
            megdnn_assert(0);
        };
        bool preferred_attribute(
                const KernSizeParam& param,
                const AlgoAttribute& positive_attr =
                        AlgoAttribute::REPRODUCIBLE,
                const AlgoAttribute& negative_attr = AlgoAttribute::DEFAULT) {
            return contain_attribute_all(positive_attr) &&
                   !contain_attribute_any(negative_attr) && preferred(param);
        };
        virtual MatmulDescription matmul_description() const = 0;

        using Mapper = std::unordered_map<AlgorithmDesc, AlgoBase*>;
    };

private:
    class AlgoF32K8x12x1;  // WASM F32 Kernel 8x12x1
    class AlgoGemv;
    class AlgoNaive;
    class AlgoPack;
    //! maintain all the algos of in the opr of wasm
    static const AlgoPack& algo_pack();
    Algorithm* get_algorithm_from_desc(const AlgorithmDesc& desc) override;

public:

    /**
     * \brief get all the algorithm for the opr.
     */
    virtual SmallVector<AlgoBase*> get_all_packed_algo();

    /**
     * \brief select algo according to input algo type
     */
    SmallVector<AlgoBase*> select_algo_type(AlgoTypePack algo_type);

protected:
    KernSizeParam make_kern_size_param(const TensorLayout& A,
                                       const TensorLayout& B,
                                       const TensorLayout& C);

    KernParam make_kern_param(_megdnn_tensor_in A, _megdnn_tensor_in B,
                              _megdnn_tensor_out C,
                              _megdnn_workspace workspace);

    std::vector<Algorithm*> get_all_algorithms(const TensorLayout& A,
                                               const TensorLayout& B,
                                               const TensorLayout& C) override;

    Algorithm* get_algorithm_heuristic(
            const TensorLayout& A, const TensorLayout& B, const TensorLayout& C,
            size_t workspace_limit_in_bytes, const AlgoAttribute& positive_attr,
            const AlgoAttribute& negative_attr) override;
};

}  // namespace wasm
}  // namespace megdnn

// vim: syntax=cpp.doxygen
