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
#include "./tensor.h"
#include <iostream>
namespace mgb::imperative::python {

interpreter::Interpreter::Channel* interpreter_for_py;

void initTensor(){
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_py = mgb::imperative::interpreter::Interpreter::inst().create_channel();
    interpreter_for_py = sl_interpreter_for_py.get();
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
    std::cout<<"make storage !"<<std::endl;
    mgb::HostTensorND ret{cn, layout.dtype};
    ret.reset(storage, layout);
    handle = interpreter_for_py->put(ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    std::cout<<"make tensor !"<<std::endl;
    return tensor;
}

std::shared_ptr<HostTensorND> makeTND(){
    auto cn = CompNode::load("xpu0");
    TensorShape shape = TensorShape{3};
    std::shared_ptr<HostTensorND> ret =
        std::make_shared<HostTensorND>(cn, shape);
    auto ptr = ret->ptr<float>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = 0.1;
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
    initTensor();
    // if (kwnames && PyTuple_GET_SIZE(kwnames)) {
    //     PyErr_SetString(PyExc_TypeError, "keyword argument not allowed");
    //     return nullptr;
    // }
    // auto* op = args[0];
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto t1 = makeTND();
    auto t2 = makeTND();
    std::cout << "creating tensor" << std::endl;
    auto a_tn = mgb::imperative::Tensor::make(*t1);
    auto b_tn = mgb::imperative::Tensor::make(*t2);
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
    for(int i = 0; i < 3; i++){
        std::cout << out_data[i] << std::endl;
    }
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