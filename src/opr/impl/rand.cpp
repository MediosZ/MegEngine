/**
 * \file src/opr/impl/rand.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/opr/rand.h"
#include "megbrain/opr/utility.h"
#include "megbrain/graph/grad_impl.h"

#include "./internal/megdnn_opr_wrapper.inl"

using namespace mgb;
using namespace opr;
using namespace intl;

template<typename MegDNNOpr>
RNGOprBase<MegDNNOpr>::RNGOprBase(const OperatorNodeBaseCtorParam &opr, const Param &param):
    Super(opr),m_param(param)
{
}

template<class MegDNNOpr>
UniqPtrWithCN<MegDNNOpr> RNGOprBase<MegDNNOpr>::create_megdnn_opr() {
    auto opr = intl::create_megdnn_opr<MegDNNOpr>(comp_node());
    opr->param() = param();
    return opr;
}

template<typename MegDNNOpr>
void RNGOprBase<MegDNNOpr>::ensure_megdnn_opr() {
    if (!m_dnn_opr || m_dnn_opr.comp_node() != comp_node()) {
        // activate comp_node for curandCreateGenerator in create_megdnn_opr
        comp_node().activate();
        m_dnn_opr = create_megdnn_opr();
    }
}

/* ================= RNG with shape =================  */
#define _INST_RNG_OPR_WITH_SHAPE(RNGOpr, name)                                      \
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNGOpr);                                                \
cg::OperatorNodeBase::NodeProp* RNGOpr::do_make_node_prop() const {                 \
    auto prop = Super::do_make_node_prop();                                         \
    prop->add_flag(NodeProp::Flag::IMPURE_FUNC);                                    \
    prop->reset_dep_type(input(), {NodeProp::DepType::HOST_VALUE});                 \
    return prop;                                                                    \
}                                                                                   \
RNGOpr::RNGOpr(VarNode *shape, const Param &param,                                  \
                        const OperatorNodeConfig &config):                          \
    Super({shape->owner_graph(), config, (name), {shape}}, param)                   \
{                                                                                   \
    DType dtype = DType::from_enum(param.dtype);                                    \
    add_input({shape});                                                             \
    add_output(None)->dtype(dtype);                                                 \
    cg::add_workspace_output(this);                                                 \
    add_equivalence_component<ScalarHash<void*>>(this);                             \
}                                                                                   \
SymbolVar RNGOpr::make(SymbolVar shape, const Param &param,                         \
                        const OperatorNodeConfig &config){                          \
    return shape.insert_single_output_opr<RNGOpr>(shape.node(), param, config);     \
}                                                                                   \
void RNGOpr::init_output_static_infer_desc() {                                      \
    using namespace cg::static_infer;                                               \
    auto &&mgr = owner_graph()->static_infer_manager();                             \
    auto infer_out = [](TensorShape &dest, const InpVal &inp) {                     \
        cg::copy_tensor_value_to_shape(dest, inp.val.at(0).value());                \
        return true;                                                                \
    };                                                                              \
    auto infer_wk = [this](TensorShape &dest, const InpVal &inp) {                  \
        ensure_megdnn_opr();                                                        \
        dest.ndim = 1;                                                              \
        dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(                          \
                {inp.val.at(0).shape(), output(0)->dtype()});                       \
        return true;                                                                \
    };                                                                              \
    mgr.register_shape_infer(output(0),                                             \
            {SourceType::DEP, {{input(0), DepType::VALUE}}, infer_out});            \
    mgr.register_shape_infer(output(1),                                             \
            {SourceType::DEP, {{output(0), DepType::SHAPE}}, infer_wk});            \
}                                                                                   \
void RNGOpr::scn_do_execute() {                                                     \
    m_dnn_opr->exec(output(0)->dev_tensor().as_megdnn(),                            \
            get_megdnn_workspace_from_var(output(1)));                              \
}

_INST_RNG_OPR_WITH_SHAPE(UniformRNG,"uniform_rng")
_INST_RNG_OPR_WITH_SHAPE(GaussianRNG,"gaussian_rng")
_INST_RNG_OPR_WITH_SHAPE(PermutationRNG,"permutation_rng")
#undef _INST_RNG_OPR_WITH_SHAPE

/* ================= RNG with input =================  */
#define _AS_MEGDNN(idx) input((idx))->dev_tensor().as_megdnn()
#define _INFER_WK_DEPS(idx) {input((idx)), DepType::SHAPE}
#define _INFER_WK_ARGS(idx) {inp.val.at((idx)).shape(), input((idx))->dtype()}

#define _INST_RNG_OPR_WITH_INPUT(RNGOpr, name)                                          \
MGB_DYN_TYPE_OBJ_FINAL_IMPL(RNGOpr);                                                    \
RNGOpr::RNGOpr(_INPUTS(VarNode*,), const Param &param,                                  \
                        const OperatorNodeConfig &config):                              \
    Super({i0->owner_graph(), config, (name), {_INPUTS(,)}}, param)                     \
{                                                                                       \
    add_input({_INPUTS(,)});                                                            \
    add_output(None)->dtype(i0->dtype());                                               \
    cg::add_workspace_output(this);                                                     \
    add_equivalence_component<ScalarHash<void*>>(this);                                 \
}                                                                                       \
SymbolVar RNGOpr::make(_INPUTS(SymbolVar,), const Param &param,                         \
                        const OperatorNodeConfig &config){                              \
    return i0.insert_single_output_opr<RNGOpr>(_INPUTS(,.node()), param, config);       \
}                                                                                       \
void RNGOpr::init_output_static_infer_desc() {                                          \
    using namespace cg::static_infer;                                                   \
    auto &&mgr = owner_graph()->static_infer_manager();                                 \
    auto infer_wk = [this](TensorShape &dest, const InpVal &inp) {                      \
        ensure_megdnn_opr();                                                            \
        dest.ndim = 1;                                                                  \
        dest.shape[0] = m_dnn_opr->get_workspace_in_bytes(                              \
                _FOR_EACH(_INFER_WK_ARGS),                                              \
                {output(0)->shape(), output(0)->dtype()});                              \
        return true;                                                                    \
    };                                                                                  \
    mgr.register_shape_infer(output(0),ShapeInferDesc::make_identity(input(0)));        \
    mgr.register_shape_infer(output(1),{SourceType::DEP, {_FOR_EACH(_INFER_WK_DEPS)},   \
                                        infer_wk});                                     \
}                                                                                       \
void RNGOpr::add_input_layout_constraint(){                                             \
    for (auto i : input()) i->add_layout_constraint_contiguous();                       \
};                                                                                      \
void RNGOpr::scn_do_execute() {                                                         \
    m_dnn_opr->exec(_FOR_EACH(_AS_MEGDNN),output(0)->dev_tensor().as_megdnn(),          \
                    get_megdnn_workspace_from_var(output(1)));                          \
}           

/* ================= 1 input =================  */
#define _INPUTS(prefix, subfix) prefix i0 subfix
#define _FOR_EACH(cb) cb(0)
_INST_RNG_OPR_WITH_INPUT(PoissonRNG,"poisson_rng")
#undef _INPUTS
#undef _FOR_EACH

/* ================= 2 input =================  */
#define _INPUTS(prefix,subfix) prefix i0 subfix, prefix i1 subfix
#define _FOR_EACH(cb) cb(0), cb(1)
_INST_RNG_OPR_WITH_INPUT(BetaRNG,"beta_rng")
_INST_RNG_OPR_WITH_INPUT(GammaRNG,"gamma_rng")
#undef _INPUTS
#undef _FOR_EACH

#undef _AS_MEGDNN
#undef _INFER_WK_DEPS
#undef _INFER_WK_ARGS
#undef _INST_RNG_OPR_WITH_INPUT

#define IMPL(_cls)                                      \
    MGB_IMPL_OPR_GRAD(_cls) {                           \
        MGB_MARK_USED_VAR(out_grad);                    \
        return InvalidGrad::make(opr, wrt_idx);         \
    }

namespace mgb {
namespace opr {
namespace intl {
template class RNGOprBase<::megdnn::GaussianRNG>;
template class RNGOprBase<::megdnn::UniformRNG>;
template class RNGOprBase<::megdnn::GammaRNG>;
template class RNGOprBase<::megdnn::PermutationRNG>;
template class RNGOprBase<::megdnn::BetaRNG>;
template class RNGOprBase<::megdnn::PoissonRNG>;
#if MGB_ENABLE_GRAD
IMPL(GaussianRNG);
IMPL(UniformRNG);
IMPL(GammaRNG);
IMPL(PoissonRNG);
IMPL(PermutationRNG);
IMPL(BetaRNG);
#endif
}             
}
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}

