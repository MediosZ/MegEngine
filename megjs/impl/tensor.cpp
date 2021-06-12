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

#include "./tensor.h"
#include <iostream>
namespace mgb::imperative::python {

interpreter::Interpreter::Channel* interpreter_for_py;

void initTensor(){
    imperative::Tensor::static_initialize();
    static auto sl_interpreter_for_py = mgb::imperative::interpreter::Interpreter::inst().create_channel();
    interpreter_for_py = sl_interpreter_for_py.get();
}

void makeTensor(){
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
    auto tensor = std::make_shared<mgb::imperative::python::Tensor>(handle);
    std::cout<<"make tensor !"<<std::endl;
}

/*
TensorWrapper::TensorWrapper(PyObject* args, PyObject* kwargs) {
    if (kwargs && PyDict_Size(kwargs)) {
        throw py::type_error("keyword argument not allowed");
    }
    auto nargs = PyTuple_Size(args);
    auto tup = py::reinterpret_borrow<py::tuple>(args);
    if (nargs == 0) {
        throw py::type_error("too few arguments");
    }
    if (auto* t = try_cast(tup[0].ptr())) {
        if (nargs > 1) {
            throw py::type_error("expect 1 argument");
        }
        m_tensor = t->m_tensor;
    } else {
        if (nargs == 1) {
            auto arg0 = PyTuple_GetItem(args, 0);
            // for lazy_eval_tensor
            if (strstr(arg0->ob_type->tp_name, "VarNode")) {
                if (PyObject_HasAttrString(arg0, "_node")) {
                    arg0 = PyObject_GetAttrString(arg0, "_node");
                }
                m_tensor = std::make_shared<Tensor>(py::handle(arg0).cast<cg::VarNode *>());
            } else {
                // for DeviceTensorND
                if (strstr(arg0->ob_type->tp_name, "DeviceTensorND")) {
                    auto dv = py::handle(arg0).cast<DeviceTensorND>();
                    interpreter::Interpreter::Handle handle = interpreter_for_py->put(dv);
                    m_tensor = std::make_shared<Tensor>(handle);
                } else {
                    throw py::type_error("single argument is not tensor, varnode or devicetensor");
                }
            }
        } 
        else {

        }
    }
}

*/
}