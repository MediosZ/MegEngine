/**
 * \file imperative/python/src/tensor.h
 * MegEngine is Licensed under the Apache License, Version 2.0 (the "License")
 *
 * Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT ARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 */

#pragma once

#include <variant>

#include "megbrain/imperative/interpreter.h"
#include <string>

#include "./grad_info.h" // for struct GradInfo
#include "./trace_info.h" // for struct TraceInfo

namespace mgb::imperative::python {

extern interpreter::Interpreter::Channel* interpreter_for_py;

class SharedHandle {
    using Handle = interpreter::Interpreter::Handle;
    static_assert(std::is_pointer_v<Handle>);
    std::shared_ptr<std::remove_pointer_t<Handle>> holder;

public:
    inline explicit SharedHandle(Handle handle) : holder(handle, [](auto* h){
        if (h) {
            interpreter_for_py->del(h);
        }
    }) {}
    SharedHandle(const SharedHandle&) = default;
    SharedHandle& operator=(const SharedHandle&) = default;
    SharedHandle(SharedHandle&&) = default;
    SharedHandle& operator=(SharedHandle&&) = default;

    inline Handle get() {return holder.get();}
};


struct Tensor : std::enable_shared_from_this<Tensor>, NonCopyableObj {
    using flags_t = uint64_t;

    struct Flags {
        static constexpr flags_t SCALAR = 1;
        static constexpr flags_t GRAD = 1 << 1;
        static constexpr flags_t TRACE = 1 << 2;
    };

    flags_t m_flags = 0;

    GradInfo m_grad_info;
    TraceInfo m_trace_info;
    SharedHandle m_handle;
    std::string user_custom_name;
    std::string automatic_name;
    cg::VarNode* m_var;

    using Handle = interpreter::Interpreter::Handle;

    inline Tensor() : m_handle(nullptr), m_var(nullptr) {}
    inline explicit Tensor(Handle handle) : m_handle(handle), m_var(nullptr) {}
    inline explicit Tensor(SharedHandle handle) : m_handle(std::move(handle)), m_var(nullptr) {}
    inline explicit Tensor(cg::VarNode *var) : m_handle(nullptr), m_var(var) {}

    ~Tensor() = default;

    inline std::shared_ptr<Tensor> copy() {
        auto ret = std::make_shared<Tensor>(m_handle);
        ret->m_flags = m_flags;
        ret->m_grad_info = m_grad_info;
        ret->m_trace_info = m_trace_info;
        ret->m_var = m_var;
        return ret;
    }

    inline DType dtype() {
        if (m_var) {
            return m_var->dtype();
        }
        return interpreter_for_py->get_dtype(m_handle.get());
    }
    inline CompNode comp_node() {
        if (m_var) {
            return m_var->comp_node();
        }
        return interpreter_for_py->get_device(m_handle.get());
    }
    inline TensorShape shape() {
        if (m_var) {
            return m_var->shape();
        }
        return interpreter_for_py->get_shape(m_handle.get());
    }

    inline HostTensorND value(){
        return interpreter_for_py->get_value(m_handle.get());
    }
};


struct ApplyContext {
    static Tensor::flags_t global_disable;

    Tensor::flags_t flags;
    std::shared_ptr<OpDef> op;
    Tensor*const* args;
    size_t nargs;
    // PyTypeObject* pytype = nullptr;
    bool backward = false;

    class scoped_disable : NonCopyableObj {
        Tensor::flags_t saved_flags;

    public:
        scoped_disable(Tensor::flags_t flags) : saved_flags(ApplyContext::global_disable) {
            ApplyContext::global_disable |= flags;
        }
        ~scoped_disable() {
            ApplyContext::global_disable = saved_flags;
        }
    };
};

using apply_result_t = SmallVector<std::shared_ptr<Tensor>, 8>;

apply_result_t apply(ApplyContext& ctx);

std::shared_ptr<Tensor> makeTensor();

void jsapply();

}