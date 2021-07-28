#include "./utils.h"

using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {

DType getDataType(int d){
    switch (d)
    {
    case 0:
        return dtype::Float32();
        break;
    case 1:
        return dtype::Int32();
        break;
    case 2:
        return dtype::Int8();
        break;
    case 3:
        return dtype::Uint8();
        break;
    default:
        return dtype::Float32();
        break;
    }
}


HostTensorND makeGrad(const TensorShape& shape){
    auto cn = CompNode::load("cpu0");
    HostTensorND ret = HostTensorND(cn, shape);

    auto ptr = ret.ptr<float>();
    for (size_t i = 0, it = shape.total_nr_elems(); i < it; ++ i) {
        ptr[i] = 1.0;
    } 
    return ret;
}


void printTensor(std::shared_ptr<Tensor> t){
    auto t_out_value = t->value().ptr<float>();
    for(size_t i = 0; i < t->shape().total_nr_elems(); i++){
        std::cout << t_out_value[i] << std::endl;
    }
}


std::shared_ptr<Tensor> make_shape(std::initializer_list<int32_t> init_shape) {
    auto cn = CompNode::load("cpu0");
    HostTensorND scalar{cn, {{init_shape.size()}, dtype::Int32()}};
    auto ptr = scalar.ptr<int32_t>();
    auto idx = 0;
    for(auto value : init_shape){
        ptr[idx] = value;
        ++idx;
    }
    Interpreter::Handle handle = interpreter_for_js->put(scalar, false);
    return std::make_shared<Tensor>(handle);
}

#ifdef __EMSCRIPTEN__

void assignData(const emscripten::val &data, std::shared_ptr<HostTensorND> ret, const int type){
    const auto ldata = data["length"].as<unsigned>();
    if(type == 0){
        auto ptr = ret->ptr<float>();
        emscripten::val dataMemoryView{emscripten::typed_memory_view(ldata, ptr)};
        dataMemoryView.call<void>("set", data);
    }
    else if(type == 1){
        auto ptr = ret->ptr<int32_t>();
        emscripten::val dataMemoryView{emscripten::typed_memory_view(ldata, ptr)};
        dataMemoryView.call<void>("set", data);
    }
    else if(type == 2){
        auto ptr = ret->ptr<int8_t>();
        emscripten::val dataMemoryView{emscripten::typed_memory_view(ldata, ptr)};
        dataMemoryView.call<void>("set", data);
    }
    else if(type == 3){
        auto ptr = ret->ptr<uint8_t>();
        emscripten::val dataMemoryView{emscripten::typed_memory_view(ldata, ptr)};
        dataMemoryView.call<void>("set", data);
    }
    else{
        throw std::runtime_error("Unknown type");
    }
}


SmallVector<size_t> getVectorFromVal(const emscripten::val &v){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);
    return rv;
}


#endif 


}
