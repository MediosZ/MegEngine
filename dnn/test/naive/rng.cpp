/**
 * \file dnn/test/naive/rng.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "megdnn.h"
#include "test/naive/fixture.h"
#include "test/naive/rng.h"
#include "test/common/tensor.h"

namespace megdnn {

namespace test {

template<typename ctype>
void assert_uniform_correct(const ctype *src, size_t size) {
    for (size_t i = 0; i < size; ++ i) {
        ASSERT_GT(src[i], ctype(0));
        ASSERT_LE(src[i], ctype(1));
    }
    auto stat = get_mean_var(src, size, ctype(0.5));
    ASSERT_LE(std::abs(stat.first - 0.5), 1e-3);
    ASSERT_LE(std::abs(stat.second - 1.0 / 12), 1e-3);
}

namespace {
    template<typename dtype>
    void run_uniform(Handle *handle) {
        auto opr = handle->create_operator<UniformRNG>();
        opr->param().dtype = DTypeTrait<dtype>::enumv;
        Tensor<typename DTypeTrait<dtype>::ctype> t(
                handle, {TensorShape{200000}, dtype()});
        opr->exec(t.tensornd(), {});
        assert_uniform_correct(t.ptr(), t.layout().total_nr_elems());
    }

    template<typename dtype>
    void run_gaussian(Handle *handle) {
        using ctype = typename DTypeTrait<dtype>::ctype;
        auto opr = handle->create_operator<GaussianRNG>();
        opr->param().mean = 0.8;
        opr->param().std = 2.3;
        opr->param().dtype = DTypeTrait<dtype>::enumv;
        Tensor<ctype> t(handle, {TensorShape{200001}, dtype()});
        opr->exec(t.tensornd(), {});

        auto ptr = t.ptr();
        auto size = t.layout().total_nr_elems();
        for (size_t i = 0; i < size; ++ i) {
            ASSERT_LE(std::abs(ptr[i] - 0.8), ctype(15));
        }
        auto stat = get_mean_var(ptr, size, ctype(0.8));

        ASSERT_LE(std::abs(stat.first - 0.8), 5e-3);
        ASSERT_LE(std::abs(stat.second - 2.3 * 2.3), 5e-2);   
    }

    template<typename dtype>
    void run_gamma(Handle* handle){

        using ctype = typename DTypeTrait<dtype>::ctype;
        auto opr = handle->create_operator<GammaRNG>();

        TensorLayout ly{TensorShape{2000000*5}, dtype()};

        Tensor<ctype> out(handle, ly);
        Tensor<ctype> shape(handle, ly);
        Tensor<ctype> scale(handle, ly);

        auto shape_ptr = shape.ptr();
        auto scale_ptr = scale.ptr();
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 2000000; ++j) {
                shape_ptr[i * 2000000 + j] = 2 * 0.3 * i + 0.5;
                scale_ptr[i * 2000000 + j] = i * 0.2 + 0.1;
            }
        }
        opr->exec(shape.tensornd(), scale.tensornd(), out.tensornd(), {});

        auto ptr = out.ptr();
        for(int i = 0; i < 5 ; ++i){
            float a = 2 * 0.3 * i + 0.5, b = i * 0.2 + 0.1;
            float mean = a * b;
            float std = a * (b * b) ;
            auto stat = get_mean_var(ptr + i * 2000000, 2000000, ctype(mean));
            ASSERT_LE(std::abs(stat.first - mean), 0.01);
            ASSERT_LE(std::abs(stat.second - std), 0.01);
        }
    }

    template<typename dtype>
    void run_poisson(Handle* handle){

        using ctype = typename DTypeTrait<dtype>::ctype;
        auto opr = handle->create_operator<PoissonRNG>();

        TensorLayout ly{TensorShape{200000*5}, dtype()};

        Tensor<ctype> out(handle, ly);
        Tensor<ctype> lam(handle, ly);

        auto lam_ptr = lam.ptr();
        for(int i = 0; i < 5; ++i){
            for(int j = 0; j <200000; ++j){
                lam_ptr[i*200000 + j] = ctype(i + 1);
            }
        }
        opr->exec(lam.tensornd(), out.tensornd(), {});

        auto ptr = out.ptr();
        for(int i = 0; i < 5 ; ++i){
            auto stat = get_mean_var(ptr + i*200000, 200000, ctype(i + 1));
            ASSERT_LE(std::abs(stat.first -   ctype(i + 1)), 0.01);
            ASSERT_LE(std::abs(stat.second -  ctype(i + 1)), 0.01);
        }
    }

    template<typename dtype>
    void run_beta(Handle* handle){

        using ctype = typename DTypeTrait<dtype>::ctype;
        auto opr = handle->create_operator<BetaRNG>();

        TensorLayout ly{TensorShape{200000*5}, dtype()};

        Tensor<ctype> out(handle, ly);
        Tensor<ctype> alpha(handle, ly);
        Tensor<ctype> beta(handle, ly);

        auto alpha_ptr = alpha.ptr();
        auto beta_ptr = beta.ptr();
        for (int i = 0; i < 5; ++i) {
            for (int j = 0; j < 200000; ++j) {
                alpha_ptr[i * 200000 + j] = 0.3 * i + 0.1;
                beta_ptr[i * 200000 + j] = 2 * i * 0.3 + 0.1;
            }
        }
        opr->exec(alpha.tensornd(),beta.tensornd(), out.tensornd(), {});

        auto ptr = out.ptr();
        for(int i = 0; i < 5 ; ++i){
            float a = 0.3 * i + 0.1, b = 2 * i * 0.3 + 0.1;
            float mean = a / (a + b);
            float std = a * b / ((a + b) * (a + b) * (a + b + 1));
            auto stat = get_mean_var(ptr + i * 200000, 200000, ctype(mean));
            ASSERT_LE(std::abs(stat.first - mean), 0.01);
            ASSERT_LE(std::abs(stat.second - std), 0.01);
        }
    }

    template<typename dtype>
    void run_permutation(Handle* handle){

        using ctype = typename DTypeTrait<dtype>::ctype;
        size_t sample_num = std::min(200000,
                    static_cast<int>(DTypeTrait<dtype>::max()) - 10);

        auto opr = handle->create_operator<PermutationRNG>();
        opr->param().dtype = DTypeTrait<dtype>::enumv;
        TensorLayout ly{TensorShape{sample_num}, dtype()};
        Tensor<ctype> t(handle, ly);
        opr->exec(t.tensornd(), {});

        auto ptr = t.ptr();
        auto size = t.layout().total_nr_elems();

        std::vector<ctype> res(size);
        int not_same = 0;
        for(size_t i = 0; i < size; ++i){
            if ((ptr[i] - ctype(i)) >= 1 ) not_same++;
            res[i] = ptr[i];
        }
        ASSERT_GT(not_same, 5000);
        std::sort(res.begin(),res.end());
        for(size_t i = 0; i < size; ++i){
            ASSERT_LE(std::abs(res[i] - ctype(i)), 1e-8);
        }
    }
}

TEST_F(NAIVE, UNIFORM_RNG_F32) {
    run_uniform<dtype::Float32>(handle());
}

TEST_F(NAIVE, UNIFORM_RNG_F16) {
    DNN_INC_FLOAT16(run_uniform<dtype::Float16>(handle()));
}

TEST_F(NAIVE, GAUSSIAN_RNG_F32) {
    run_gaussian<dtype::Float32>(handle());
}

TEST_F(NAIVE, GAUSSIAN_RNG_F16) {
    DNN_INC_FLOAT16(run_gaussian<dtype::Float16>(handle()));
}

TEST_F(NAIVE, GAMMA_RNG_F32) {
    run_gamma<dtype::Float32>(handle());
}

TEST_F(NAIVE, GAMMA_RNG_F16) {
    DNN_INC_FLOAT16(run_gamma<dtype::Float16>(handle()));
}

TEST_F(NAIVE, POISSON_RNG_F32) {
    run_poisson<dtype::Float32>(handle());
}

TEST_F(NAIVE, POISSON_RNG_F16) {
    DNN_INC_FLOAT16(run_poisson<dtype::Float16>(handle()));
}

TEST_F(NAIVE, BETA_RNG_F32) {
    run_beta<dtype::Float32>(handle());
}

TEST_F(NAIVE, BETA_RNG_F16) {
    DNN_INC_FLOAT16(run_beta<dtype::Float16>(handle()));
}

TEST_F(NAIVE, PERMUTATION_RNG_F32) {
    run_permutation<dtype::Float32>(handle());
}

TEST_F(NAIVE, PERMUTATION_RNG_INT32) {
    run_permutation<dtype::Int32>(handle());
}

TEST_F(NAIVE, PERMUTATION_RNG_INT16) {
    run_permutation<dtype::Int16>(handle());
}

} // namespace test
} // namespace megdnn

// vim: syntax=cpp.doxygen



