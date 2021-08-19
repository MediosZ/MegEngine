/**
 * \file dnn/test/wasm/elemwise.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */
#include "test/common/elemwise.h"
#include "megdnn/oprs.h"
#include "test/common/checker.h"
#include "test/common/rng.h"
#include "test/wasm/fixture.h"

using namespace megdnn;
using namespace test;

void print4D(const TensorND& tensor) {
    TensorLayout layout = tensor.layout;
    float* result = tensor.ptr<float>();
    size_t N = layout.shape[0], C = layout.shape[1], H = layout.shape[2],
           W = layout.shape[3];
    size_t it = 0;
    rep(n, N) {
        rep(c, C) {
            rep(h, H) {
                rep(w, W) { printf("%.4f ", result[it++]); }
                printf("\n");
            }
            printf("\n");
        }
        printf("\n");
    }
}

#define UNARY_TEST_CASE(_optr)                                \
    checker.set_param(Mode::_optr).execs({{1, 1556011}, {}}); \
    checker.set_param(Mode::_optr).execs({{1, 7}, {}});

#define BUILD_UNARY_TEST_CASE_INT \
    UNARY_TEST_CASE(RELU)         \
    UNARY_TEST_CASE(ABS)

#define BUILD_UNARY_TEST_CASE_FLOAT \
    UNARY_TEST_CASE(ABS)            \
    UNARY_TEST_CASE(LOG)            \
    UNARY_TEST_CASE(COS)            \
    UNARY_TEST_CASE(SIN)            \
    UNARY_TEST_CASE(FLOOR)          \
    UNARY_TEST_CASE(CEIL)           \
    UNARY_TEST_CASE(SIGMOID)        \
    UNARY_TEST_CASE(EXP)            \
    UNARY_TEST_CASE(TANH)           \
    UNARY_TEST_CASE(RELU)           \
    UNARY_TEST_CASE(ROUND)

TEST_F(WASM, ELEMWISE_FORWARD_UNARY) {
    using Mode = ElemwiseForward::Param::Mode;
    Checker<ElemwiseForward> checker(handle());
    // case int
    checker.set_dtype(0, dtype::Int8());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int16());
    BUILD_UNARY_TEST_CASE_INT

    checker.set_dtype(0, dtype::Int32());
    BUILD_UNARY_TEST_CASE_INT

    // case float
    UniformFloatRNG rng(1e-2, 6e1);
    checker.set_rng(0, &rng);
    checker.set_epsilon(1e-6);
    checker.set_dtype(0, dtype::Float32());
    BUILD_UNARY_TEST_CASE_FLOAT
}


template <typename tag>
class WASM_ELEMWISE : public WASM {};
TYPED_TEST_CASE(WASM_ELEMWISE, elemwise::test_types);
TYPED_TEST(WASM_ELEMWISE, run) {
    elemwise::run_test<TypeParam>(this->handle());
}

// vim: syntax=cpp.doxygen
