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
#include <iostream>
#include <vector>

template <typename T>
T prepare_backward_graph_inputs(const mgb::imperative::BackwardGraphResult& bg, const T& inputs, const T& outputs, const T& grads) {
    T ret;
    size_t i = 0;
    for (auto&& t : inputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : outputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : grads) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    return ret;
}

template <typename T, typename U>
T expand_grads(const U& bg, const T& outputs) {
    T ret(bg.input_has_grad.size());
    for (size_t i = 0, j = 0; i < bg.input_has_grad.size(); ++i) {
        if (bg.input_has_grad[i]) {
            ret[i] = outputs[j++];
        }
    }
    return ret;
}

template <typename T>
T prepare_optimized_backward_inputs(const mgb::imperative::OptimizedBackwardGraphResult& bg, const T& precomp, const T& inputs, const T& outputs, const T& grads) {
    T ret = precomp;
    size_t i = 0;
    for (auto&& t : inputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : outputs) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    for (auto&& t : grads) {
        if (bg.save_for_backward[i++]) {
            ret.push_back(t);
        }
    }
    return ret;
}


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

std::shared_ptr<Tensor> gen_randn(Tensor* x) {
    static auto op = GaussianRNG::make();
    return js::apply(op, x)[0];
}

std::shared_ptr<Tensor> make_tensor_like(Tensor* other, float v = 0) {
    HostTensorND scalar{other->comp_node(), {{1}, dtype::Float32()}};
    scalar.ptr<float>()[0] = v;
    interpreter::Interpreter::Handle handle = interpreter_for_js->put(scalar, false);
    auto&& t = std::make_shared<Tensor>(handle);
    auto res = broadcast_to(t.get(), get_shape(other).get());
    return res;
}


std::shared_ptr<Tensor> makeTensor(){
    dt_float32* data = new dt_float32[3];
    data[0] = 0.1;
    data[1] = 0.2;
    data[3] = 0.3;

    mgb::imperative::interpreter::Interpreter::Handle handle;
    DType dtype = DType::from_enum(DTypeEnum::Float32);
    CompNode cn = CompNode::load("c pux");
    
    TensorLayout layout;
    layout.dtype = dtype;
    layout.ndim = 1;
    layout.shape[0] = 3;
    layout.stride[0] = 4;

    auto storage = mgb::HostTensorStorage(cn);
    storage.ensure_size(layout.span().dist_byte());
    memcpy(storage.ptr(), data, layout.span().dist_byte());

    mgb::HostTensorND ret{cn, layout.dtype};
    ret.reset(storage, layout);
    handle = interpreter_for_js->put(ret, true);
    auto tensor = std::make_shared<Tensor>(handle);

    return tensor;
}

std::shared_ptr<HostTensorND> makeTNDI(const TensorShape& shape){
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());

    auto ptr = ret->ptr<int32_t>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = 2 + i;
    }
    return ret;
}

std::shared_ptr<HostTensorND> makeTND(const TensorShape& shape, bool grad = false){
    auto cn = CompNode::load("cpu0");
    // TensorShape shape = TensorShape{3};
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);

    auto ptr = ret->ptr<float>();
    if(grad){
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
            ptr[i] = 1.0;
        } 
    }
    else{
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
            ptr[i] = 0.1 * i;
        }
    }
    return ret;
}

HostTensorND* makeTNDP(const TensorShape& shape, bool grad = false){
    auto cn = CompNode::load("cpu0");
    // TensorShape shape = TensorShape{3};
    // std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    HostTensorND* ret = new HostTensorND(cn, shape);
    auto ptr = ret->ptr<float>();
    if(grad){
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
            ptr[i] = 1.0;
        } 
    }
    else{
        for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
            ptr[i] = 0.1 * i;
        }
    }
    return ret;
}


Tensor::flags_t ApplyContext::global_disable = 0;


apply_result_t apply(ApplyContext& ctx) {
    // emulating scalar should be put to specific op's apply, e.g.,
    // elementwise, reduce, typecvt. Currently it's still handled at python
    // side. It could be move to C++ side if it has an impact on performance
    auto flags = ctx.flags & ~ApplyContext::global_disable;
    if (flags & Tensor::Flags::SCALAR) {
        // TODO: emulate scalar
    }

    if (flags & Tensor::Flags::GRAD) {
        return apply_grad(ctx);
    }

    SmallVector<interpreter::Interpreter::Handle> handles(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        handles[i] = ctx.args[i]->m_handle.get();
    }
    auto output_handles = interpreter_for_js->apply_op(ctx.op, handles);

    apply_result_t outputs;
    outputs.reserve(output_handles.size());
    for (auto h : output_handles) {
        outputs.emplace_back(std::make_shared<Tensor>(h));
    }
    return outputs; 

    // mgb_assert(0);
}

extern "C"{

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void initTensor(){
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_js = mgb::imperative::interpreter::Interpreter::inst().create_channel();
    interpreter_for_js = sl_interpreter_for_js.get();
}


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* registerTensor(const size_t tensor_id, const size_t shapeSize, size_t* shape_offset) {
    SmallVector<size_t> shapeVec(shapeSize);
    for (int i = 0; i < shapeSize; i++){
        shapeVec[i] = shape_offset[i];
    }
    TensorShape shape = TensorShape(shapeVec);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    auto handle = interpreter_for_js->put(*ret, true);
    return handle;
}


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* randn(const size_t shapeSize, size_t* shape_offset){
    auto cn = CompNode::load("cpu0");
    TensorShape shape = TensorShape{shapeSize};
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = shape_offset[i];
    }
    auto handle = interpreter_for_js->put(*ret, true);
    SmallVector<mgb::imperative::interpreter::Interpreter::Handle> handleVec(1);
    handleVec[0] = handle;
    auto op = std::shared_ptr<OpDef>(GaussianRNG::make());
    auto outhandles = interpreter_for_js->apply_op(op, handleVec); 
    return outhandles[0];
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* add(void* aId, void* bId){
    auto a = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(aId);
    auto b = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(bId);
    SmallVector<mgb::imperative::interpreter::Interpreter::Handle> handleVec(2);
    handleVec[0] = a;
    handleVec[1] = b;
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto outhandles = interpreter_for_js->apply_op(op, handleVec);
    return outhandles[0];
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* mul(void* aId, void* bId){
    auto a = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(aId);
    auto b = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(bId);
    SmallVector<mgb::imperative::interpreter::Interpreter::Handle> handleVec(2);
    handleVec[0] = a;
    handleVec[1] = b;
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::MUL));
    auto outhandles = interpreter_for_js->apply_op(op, handleVec);
    return outhandles[0];
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* getOffset(void* id){
   auto handle = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(id);
   auto tensor = interpreter_for_js->get_value(handle);
   auto ptr = tensor.ptr<float>();
   return ptr;
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void jsbackward(){
    mgb::set_log_level(mgb::LogLevel::INFO);
    mgb_log("Test Backward");
    auto gradKey = std::make_shared<GradKey>();
    ApplyContext ctx;
    ctx.flags = 0;
    ctx.op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::MUL));
    SmallVector<Tensor*, 64> tensors(2);
    ctx.args = &tensors[0];
    ctx.nargs = 2;
    if (ctx.op->same_type<BackwardGraph>()) {
        ctx.backward = true;
    }
    auto ht1 = makeTND({5});
    auto ht2 = makeTND({5});
    auto t1 = interpreter_for_js->put(*ht1, true);
    auto t2 = interpreter_for_js->put(*ht2, true);
    std::shared_ptr<Tensor> pt1 = std::make_shared<Tensor>(t1);
    std::shared_ptr<Tensor> pt2 = std::make_shared<Tensor>(t2);
    gradKey->attach(pt1.get());
    gradKey->attach(pt2.get());
    
    ctx.flags |= pt1->m_flags;
    ctx.flags |= pt2->m_flags;

    tensors[0] = pt1.get();
    tensors[1] = pt2.get();
    auto res = apply(ctx);
    HostTensorND outTensor = res[0]->value();
    mgb_log("Forward:");
    auto t_out_data = outTensor.ptr<float>();
    for(int i = 0; i < 5; i++){
        mgb_log("Value: %f", t_out_data[i]);
    }

    std::vector<Tensor*> ts;
    ts.emplace_back(res[0].get());
    // z = x + y;
    // dz/dx = 1;
    // dz/dy = 1;
    std::vector<Tensor*> gs;
    // auto pdy = make_tensor_like(res[0].get(), 1.0);
    auto dy = interpreter_for_js->put(*makeTND({5}, true), true);
    // std::shared_ptr<Tensor> pdy = std::make_shared<Tensor>(dy);
    // don't use smart_pointer, will cause double-free
    Tensor* pdy = new Tensor(dy);
    gs.emplace_back(pdy);
    mgb_log("Backward:");
    gradKey->backward(ts, gs);
    mgb_log("done!");
    auto sop = GetVarShape::make();
    auto sha = js::apply(sop, pt1.get())[0];
    auto t_out_shape = sha->value().ptr<int32_t>();
    for(int i = 0; i < sha->shape().total_nr_elems(); i++){
        mgb_log("Shape: %d", t_out_shape[i]);
    }

    interpreter_for_js->sync();
    
}

void testRand(){
    mgb_log("Test Rand");
    size_t s[2] = {3, 4};
    auto cn = CompNode::load("cpux");
    TensorShape shape = TensorShape{2};
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = s[i];
    }
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    // auto op = std::shared_ptr<OpDef>(GaussianRNG::make());
    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto out = js::apply(op, tensor.get(), tensor.get())[0];

/*     auto t_out_value = out->value().ptr<float>();
    for(int i = 0; i < out->shape().total_nr_elems(); i++){
        std::cout << t_out_value[i] << std::endl;
    } */


    interpreter_for_js->sync();

}


} // extern C


void jsapply() {
    //auto op = Elemwise::make(Elemwise::Mode::ADD);
    auto op = std::shared_ptr<OpDef>(GaussianRNG::make());
    auto t1 = makeTNDI({2});
    // auto t2 = makeTNDI({2});
    auto handle1 = interpreter_for_js->put(*t1, false);
    // auto handle2 = interpreter_for_js->put(*t2, false);

    SmallVector<mgb::imperative::interpreter::Interpreter::Handle> handleVec(1);
    handleVec[0] = handle1;
    // handleVec[1] = handle2;
    
    auto outhandles = interpreter_for_js->apply_op(op, handleVec);
    // std::cout << getOffset(outhandles[0]) << std::endl;
    // be careful with get_value!
    auto outTensor = interpreter_for_js->get_value(outhandles[0]);
    // you should delete it manually. but why?
    interpreter_for_js->del(outhandles[0]);

    auto t_out_data = outTensor.ptr<float>();
    for(int i = 0; i < 6; i++){
        std::cout << t_out_data[i] << std::endl;
    }

    interpreter_for_js->sync();

}


} // namespace

#ifndef __EMSCRIPTEN__

int main(){
    mgb_log("main function");
    mgb::set_log_level(mgb::LogLevel::INFO);
    mgb::imperative::js::initTensor();
    mgb::imperative::js::jsapply();
    // mgb::imperative::js::jsbackward();
    // mgb::imperative::js::testRand();

}

#endif