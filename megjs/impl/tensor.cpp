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
#include "megbrain/imperative/webassembly.h"
#include "./tensor.h"
#include <iostream>


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


namespace mgb::imperative::python {

interpreter::Interpreter::Channel* interpreter_for_py;

extern "C"{

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void initTensor(){
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_py = mgb::imperative::interpreter::Interpreter::inst().create_channel();
    interpreter_for_py = sl_interpreter_for_py.get();
}


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* registerTensor(const size_t tensor_id, 
    const size_t size, void* memory_offset,
    const size_t shapeSize, size_t* shape_offset) {
    // std::cout << "tensor id: " << tensor_id << " size: "<< size << std::endl;
    float* data = reinterpret_cast<float*>(memory_offset);
    
    SmallVector<size_t> shapeVec(shapeSize);
    int* inshape = reinterpret_cast<int*>(shape_offset);
    for (int i = 0; i < shapeSize; i++){
        // std::cout << "shape: " << inshape[i] << std::endl;
        shapeVec[i] = inshape[i];
    }
    auto cn = CompNode::load("xpu0");
    TensorShape shape = TensorShape(shapeVec);
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    
    auto ptr = ret->ptr<float>();
    // std::cout << "make tensor with size: " << shape.total_nr_elems() << std::endl;
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = data[i];
        // std::cout << "data: " << ptr[i] << std::endl;
    }
    

    auto handle = interpreter_for_py->put(*ret, true);
    return handle;
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
    
    auto outhandles = interpreter_for_py->apply_op(op, handleVec);
    return outhandles[0];
}

#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void* getOffset(void* id){
   auto handle = reinterpret_cast<mgb::imperative::interpreter::Interpreter::Handle>(id);
   auto tensor = interpreter_for_py->get_value(handle);
   auto ptr = tensor.ptr<float>();
   return ptr;
}

}
std::shared_ptr<Tensor> makeTensor(){
    dt_float32* data = new dt_float32[3];
    data[0] = 0.1;
    data[1] = 0.2;
    data[3] = 0.3;

    mgb::imperative::interpreter::Interpreter::Handle handle;
    DType dtype = DType::from_enum(DTypeEnum::Float32);
    CompNode cn = CompNode::load("xpux");
    
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
    handle = interpreter_for_py->put(ret, true);
    auto tensor = std::make_shared<Tensor>(handle);

    return tensor;
}

std::shared_ptr<HostTensorND> makeTND(const TensorShape& shape){
    auto cn = CompNode::load("xpu0");
    // TensorShape shape = TensorShape{3};
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape);
    auto ptr = ret->ptr<float>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = 0.1 * i;
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

    //if (flags & Tensor::Flags::GRAD) {
    //    return apply_grad(ctx);
    //}

    SmallVector<interpreter::Interpreter::Handle> handles(ctx.nargs);
    for (size_t i = 0; i < ctx.nargs; ++i) {
        handles[i] = ctx.args[i]->m_handle.get();
    }

    auto output_handles = interpreter_for_py->apply_op(ctx.op, handles);

    apply_result_t outputs;
    outputs.reserve(output_handles.size());
    for (auto h : output_handles) {
        outputs.emplace_back(std::make_shared<Tensor>(h));
    }
    return outputs;

    // mgb_assert(0);
}


void jsapply() {
    // initTensor();
    // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
    //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
    //     return nullptr;
    // }
    // auto* op = args[0];
    auto cn = CompNode::load("xpu0");
    LogicalTensorDesc desc = {TensorLayout(dtype::Float32()), cn};
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto bg = OpDef::make_backward_graph(*op, {desc, desc}, {true, true}, {true});
    auto obg = OptimizedBackwardGraphResult(bg);

    auto t1 = makeTND({5});
    auto t2 = makeTND({5});
    auto t3 = makeTND({5});


    std::cout << "creating tensor" << std::endl;
    auto a_tn = mgb::imperative::Tensor::make(*t1);
    auto b_tn = mgb::imperative::Tensor::make(*t2);
    auto dc_tn = mgb::imperative::Tensor::make(*t3);

    std::cout << "before apply" << std::endl;
    // auto c_tn = OpDef::apply_on_physical_tensor(*op, {a_tn, b_tn})[0];


    auto inputs = SmallVector<TensorPtr>{a_tn, b_tn};
    SmallVector<DeviceTensorND> inp_tensornds(inputs.size());
    for (unsigned i = 0; i < inputs.size(); ++i){
        inp_tensornds[i] = inputs[i]->dev_tensor();
    }
    SmallVector<DeviceTensorND> oup_tensornds = {{inp_tensornds[0].comp_node(), inp_tensornds[0].dtype()}};
    // apply_on_device_tensornd(def, inp_tensornds, &oup_tensornds);
    // return {Tensor::make(oup_tensornds[0])};
    OpDef::apply_on_device_tensornd(*op, inp_tensornds, &oup_tensornds);
    auto c_tn = mgb::imperative::Tensor::make(oup_tensornds[0]);


    std::cout << "after apply" << std::endl;
    auto out = c_tn->get_value();
    auto out_data = out.ptr<float>();
    for(int i = 0; i < 5; i++){
        std::cout << out_data[i] << std::endl;
    }

    auto backward_graph_inputs = prepare_backward_graph_inputs<SmallVector<TensorPtr>>(bg, {a_tn, b_tn}, {c_tn}, {dc_tn});
    auto grads = expand_grads(bg, OpDef::apply_on_physical_tensor(*bg.backward, backward_graph_inputs));

    auto precomp = OpDef::apply_on_physical_tensor(*obg.precomp, {a_tn, b_tn, c_tn});
    /*
    ASSERT_EQ(precomp.size(), 2);
    ASSERT_EQ(precomp[0]->shape().ndim, 1);
    ASSERT_LE(precomp[0]->shape()[0], 2);
    ASSERT_EQ(precomp[1]->shape().ndim, 1);
    ASSERT_LE(precomp[1]->shape()[0], 2);
    */
    auto backward_inputs = prepare_optimized_backward_inputs<SmallVector<TensorPtr>>(obg, precomp, {a_tn, b_tn}, {c_tn}, {dc_tn});
    auto grads2 = expand_grads(obg, OpDef::apply_on_physical_tensor(*obg.backward, backward_inputs));
    std::cout << "grads: " << grads[0]->get_value().ptr<float>()[0] << " and " << grads2[0]->get_value().ptr<float>()[0] << std::endl;
    std::cout << "grads: " << grads[1]->get_value().ptr<float>()[0] << " and " << grads2[1]->get_value().ptr<float>()[0] << std::endl;
    std::cout << "success." << std::endl;

    auto handle1 = interpreter_for_py->put(*t1, true);
    auto handle2 = interpreter_for_py->put(*t2, true);
    SmallVector<mgb::imperative::interpreter::Interpreter::Handle> handleVec(2);
    // handleVec.push_back(handle1);
    // handleVec.push_back(handle2);
    handleVec[0] = handle1;
    handleVec[1] = handle2;
    std::cout << "before do apply" << std::endl;
    auto outhandles = interpreter_for_py->apply_op(op, handleVec);
    std::cout << "output handle" << std::endl;
    HostTensorND outTensor = interpreter_for_py->get_value(outhandles[0]);
    auto tout = mgb::imperative::Tensor::make(outTensor);
    auto t_out = tout->get_value();
    auto t_out_data = t_out.ptr<float>();
    for(int i = 0; i < 5; i++){
        std::cout << t_out_data[i] << std::endl;
    }
    /*
    ASSERT_EQ(grads2.size(), 2);
    MGB_ASSERT_TENSOR_EQ(grads[0]->get_value(), grads2[0]->get_value());
    MGB_ASSERT_TENSOR_EQ(grads[1]->get_value(), grads2[1]->get_value())
    */
    /*
    SmallVector<Tensor*, 64> tensors(2);
    // tensors.push_back(t1.get());
    // tensors.push_back(t2.get());
    ApplyContext ctx;
    ctx.flags = 0;
    ctx.args = &tensors[0];
    ctx.nargs = 2;
    tensors[0] = t1.get();
    tensors[1] = t2.get();
    // ctx.pytype = pytype;
    //if (ctx.op->same_type<BackwardGraph>()) {
    //    ctx.backward = true;
    //}
    std::cout << "start calculation" << std::endl;
    // tensors 
    auto outputs = apply(ctx);
    std::cout << "end calculation" << std::endl;
    auto out = outputs[0];
    std::cout << out->shape().to_string() << std::endl;

    dt_byte* data_ptr = out->value().raw_ptr();
    auto data = data_ptr->as<float>();
    std::cout << "get data" << std::endl;
    for(int i = 0; i < 3; i++){
        std::cout << data[0] << std::endl;
    }
    std::cout << "calculation return " << std::endl;
    */
}


}