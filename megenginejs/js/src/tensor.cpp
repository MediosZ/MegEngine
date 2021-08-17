/**
 * \file imperative/python/src/tensor.cpp
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#include "megbrain/dtype.h"
#include "megbrain/common.h"
#include "megbrain/imperative/ops/utility.h"
#include "megbrain/imperative/ops/backward_graph.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/backward_graph_opt.h"
#include "./webassembly.h"
#include "./tensor.h"
#include "./grad.h"
#include "./engine.h"
#include <iostream>
#include <vector>



using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {

interpreter::Interpreter::Channel* interpreter_for_js;


bool is_tracing = false;

std::shared_ptr<Tensor> broadcast_to(Tensor* x, Tensor* s) {
    static auto op = Broadcast::make();
    return js::apply(op, x, s)[0];
}

std::shared_ptr<Tensor> get_shape(Tensor* x) {
    static auto op = GetVarShape::make();
    return js::apply(op, x)[0];
}

std::shared_ptr<Tensor> make_tensor_like(Tensor* other, float v) {
    HostTensorND scalar{other->comp_node(), {{1}, dtype::Float32()}};
    scalar.ptr<float>()[0] = v;
    Interpreter::Handle handle = interpreter_for_js->put(scalar, false);
    auto&& t = std::make_shared<Tensor>(handle);
    auto res = broadcast_to(t.get(), get_shape(other).get());
    return res;
}


std::shared_ptr<Tensor> make_const(imperative::TensorPtr value) {
    return std::make_shared<Tensor>(interpreter_for_js->put(value->dev_tensor()));
}
Tensor::flags_t ApplyContext::global_disable = 0;
Tensor::flags_t ApplyContext::global_enable = 0;

apply_result_t apply(ApplyContext& ctx) {
    // emulating scalar should be put to specific op's apply, e.g.,
    // elementwise, reduce, typecvt. Currently it's still handled at python
    // side. It could be move to C++ side if it has an impact on performance
    auto flags = ctx.flags & ~ApplyContext::global_disable;
    flags = flags | ApplyContext::global_enable;

    if (flags & Tensor::Flags::SCALAR) {
        // TODO: emulate scalar
    }

    if (flags & Tensor::Flags::GRAD) {
        return apply_grad(ctx);
    }

    SmallVector<Interpreter::Handle> handles(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        handles[i] = ctx.args[i]->m_handle.get();
    }
    
    apply_result_t outputs;

        // fast copy without really applying
        if (ctx.op->same_type<FastpathCopy>()) {
            mgb_assert(ctx.nargs == 1);
            outputs.reserve(ctx.nargs);
            outputs.emplace_back(std::make_shared<Tensor>(ctx.args[0]->m_handle));
            return outputs;
        }

        auto output_handles = interpreter_for_js->apply_op(ctx.op, handles);

    outputs.reserve(output_handles.size());
    for (auto h : output_handles) {
        outputs.emplace_back(std::make_shared<Tensor>(h));
    }
    return outputs; 

    // mgb_assert(0);
}

std::shared_ptr<Tensor> randTensor(std::initializer_list<int> init_list){
    TensorShape shape = TensorShape{init_list.size()};
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    int count{0};
    for (auto i : init_list) {
        ptr[count] = i;
        ++count;
    }
    auto seed = rand();
    auto rngHandle = rng::new_handle(cn, seed);
    auto op = GaussianRNG::make(seed, 0.0, 1.0, dtype::Float32(), rngHandle);
    
    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto result = js::apply(op, tensor.get())[0];
    interpreter_for_js->sync();
    return result;
}


void initTensor(){
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_js = Interpreter::inst().create_channel();
    interpreter_for_js = sl_interpreter_for_js.get();

    mgb::set_log_level(mgb::LogLevel::INFO);
}

void testJSBack(){
    auto wrapper = EngineWrapperInst();
    auto a = randTensor({8, 8});
    auto b = randTensor({8, 8});
    auto aid = wrapper->registerTensor(a);
    auto bid = wrapper->registerTensor(b);
    
    wrapper->startScope();
    wrapper->attach(aid);
    wrapper->attach(bid);
    auto outid = wrapper->matmul(aid, bid, false, false);
    auto addid = wrapper->add(aid, bid);
    wrapper->backward(outid);
    wrapper->endScope();
    interpreter_for_js->sync();
    // delete wrapper;
}



} // namespace

int main(){
    mgb_log("main function");
    mgb::imperative::js::initTensor();
    mgb::imperative::js::testJSBack();
}