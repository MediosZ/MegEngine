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
#include <iostream>
namespace mgb::imperative::js {
struct GradKey;
extern interpreter::Interpreter::Channel* interpreter_for_js;

class SharedHandle {
    using Handle = interpreter::Interpreter::Handle;
    static_assert(std::is_pointer_v<Handle>);
    std::shared_ptr<std::remove_pointer_t<Handle>> holder;

public:
    inline explicit SharedHandle(Handle handle) : holder(handle, [](auto* h){
        if (h) {
            interpreter_for_js->del(h);
        }
    }) {}
    SharedHandle(const SharedHandle&) = default;
    SharedHandle& operator=(const SharedHandle&) = default;
    SharedHandle(SharedHandle&&) = default;
    SharedHandle& operator=(SharedHandle&&) = default;

    inline Handle get() {return holder.get();}
};

class GradInfoCollection {
private:
    SmallVector<GradInfo> m_storage;
protected:
    void _shrink();
public:
    bool contains(GradKey* key);
    GradInfo& operator[](GradKey* key);
    GradInfo& at(GradKey* key);
    bool empty() {
        _shrink();
        return m_storage.empty();
    }
    auto begin() {
        _shrink();
        return m_storage.begin();
    }
    auto end() {
        _shrink();
        return m_storage.end();
    }
    size_t count(GradKey* key) {
        return contains(key) ? 1 : 0;
    }
};



struct Tensor : std::enable_shared_from_this<Tensor>, NonCopyableObj {
    using flags_t = uint64_t;

    struct Flags {
        static constexpr flags_t SCALAR = 1;
        static constexpr flags_t GRAD = 1 << 1;
        static constexpr flags_t TRACE = 1 << 2;
    };

    flags_t m_flags = 0;

    GradInfoCollection m_grad_info_dict;
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
        ret->m_grad_info_dict = m_grad_info_dict;
        ret->m_trace_info = m_trace_info;
        ret->m_var = m_var;
        return ret;
    }

    inline DType dtype() {
        if (m_var) {
            return m_var->dtype();
        }
        return interpreter_for_js->get_dtype(m_handle.get());
    }
    inline CompNode comp_node() {
        if (m_var) {
            return m_var->comp_node();
        }
        return interpreter_for_js->get_device(m_handle.get());
    }
    inline TensorShape shape() {
        if (m_var) {
            return m_var->shape();
        }
        return interpreter_for_js->get_shape(m_handle.get());
    }

    inline HostTensorND value(){
        return interpreter_for_js->get_value(m_handle.get());
    }
};

using CallbackFunc = std::function<void(std::shared_ptr<Tensor>)>;

struct TensorWrapper{
    TensorWrapper(int tensor) : _tensor(tensor){
        _grad = -1;
    }
    int _tensor;
    int _grad;
};


struct ApplyContext {
    static Tensor::flags_t global_disable;
    static Tensor::flags_t global_enable;

    Tensor::flags_t flags = 0;
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

template <typename T>
decltype(auto) resolve_arrow(T&& p) {
    if constexpr (std::is_pointer_v<std::remove_reference_t<T>>) {
        auto* ret = p;
        return ret;
    } else {
        auto probe = [](auto&& p) -> decltype(p.operator->()) {};
        if constexpr (std::is_invocable_v<decltype(probe), decltype(p)>) {
            return resolve_arrow(p.operator->());
        } else {
            return std::forward<T>(p);
        }
    }
}

template <typename... Args>
constexpr bool is_all_tensor_ptr = (... && std::is_same_v<decltype(resolve_arrow(std::declval<Args>())), Tensor*>);

template <typename... Args, std::enable_if_t<is_all_tensor_ptr<Args...>, int> = 0>
apply_result_t apply(std::shared_ptr<OpDef> op, Args&&... args) {
    ApplyContext ctx;
    Tensor* arg_arr[] = {resolve_arrow(args)...};
    ctx.flags = (0 | ... | args->m_flags);
    ctx.args = arg_arr;
    ctx.nargs = sizeof...(args);
    ctx.op = std::move(op);
    return apply(ctx);
}

inline auto apply(std::shared_ptr<OpDef> op, Tensor*const* args, size_t nargs) {
    ApplyContext ctx;
    ctx.op = std::move(op);
    ctx.nargs = nargs;
    ctx.args = args;
    for (size_t i = 0; i < nargs; ++i) {
        ctx.flags |= args[i]->m_flags;
    }
    return apply(ctx);
}

template <typename T>
auto apply(std::shared_ptr<OpDef> op, T&& tensors)
        -> std::enable_if_t<std::is_same_v<decltype(resolve_arrow(tensors[0])), Tensor*>,
                            apply_result_t> {
    size_t nargs = tensors.size();
    Tensor* args[nargs];
    for (size_t i = 0; i < nargs; ++i) {
        args[i] = resolve_arrow(tensors[i]);
    }
    return apply(op, args, nargs);
}

std::shared_ptr<Tensor> make_const(imperative::TensorPtr value);

inline auto apply(Subgraph graph, Tensor*const* args, size_t nargs) {
    SmallVector<std::shared_ptr<Tensor>> inputs;
    for (size_t i = 0; i < nargs; ++i) {
        inputs.push_back(args[i]->shared_from_this());
    }
    auto apply_functor = [](std::shared_ptr<OpDef> op, SmallVector<std::shared_ptr<Tensor>> inputs) {
        return apply(op, std::move(inputs));
    };
    return graph.apply(inputs, apply_functor, &make_const);
}

template <typename T>
auto apply(Subgraph graph, T&& tensors)
        -> std::enable_if_t<std::is_same_v<decltype(tensors[0]), Tensor*>,
                            apply_result_t> {
    size_t nargs = tensors.size();
    Tensor* args[nargs];
    for (size_t i = 0; i < nargs; ++i) {
        args[i] = resolve_arrow(tensors[i]);
    }
    return apply(graph, args, nargs);
}
void testJSBack();
void initTensor();
}