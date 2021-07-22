#include "./engine.h"
#include "./tensor.h"

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




Engine& Engine::inst() {
    static Engine inst_;
    return inst_;
}

Engine::Engine(){
    // mgb_log("Create Engine");
    inScope = true;
    tensor_registry = std::make_shared<std::unordered_map<int, std::shared_ptr<Tensor>>>();
}

Engine::~Engine(){
    mgb_log("%lu tensors", tensor_registry->size());
    mgb_log("Delete Engine");
}

void Engine::startScope(){
    gradkey = std::make_shared<GradKey>();
    inScope = true;
}

void Engine::endScope(){
    inScope = false;
}

void Engine::attach(Tensor* t, CallbackFunc&& callback){
    mgb_assert(inScope, "attach must be called inside a scope");
    gradkey->attach(t, std::move(callback));
}

void Engine::backward(std::vector<Tensor*> tensors, std::vector<Tensor*> grads){
    mgb_assert(inScope, "backward must be called inside a scope");
    gradkey->backward(tensors, grads);
}

int Engine::registerTensor(std::shared_ptr<Tensor> tensor){
    auto id = nextTensorID++;
    insertTensor(id, tensor);
    return id;
}

EngineWrapper::EngineWrapper(){
    _tensor_wrapper_registry = std::make_shared<std::unordered_map<int, std::shared_ptr<TensorWrapper>>>();
}

void EngineWrapper::disposeTensor(int id){
    // mgb_log("disposeTensor %d", id);
    auto tensor = getTensorWrapper(id);
    _engine.disposeTensor(tensor->_tensor);
    if(tensor->_grad == -1){
        _engine.disposeTensor(tensor->_grad);
    }
    _tensor_wrapper_registry->erase(id);
}

void EngineWrapper::attach(int32_t id){
    auto tensor_wrapper = getTensorWrapper(id);
    auto tensor = getTensor(id);
    _engine.attach(tensor.get(), [tensor_wrapper, this](std::shared_ptr<Tensor> grad){
        // there is no grad in current tensor
        if(tensor_wrapper->_grad == -1){
            auto id = _engine.registerTensor(std::move(grad));
            tensor_wrapper->_grad = id;
        }
        //replace old grad
        else{
            // mgb_log("replace grad %d", tensor_wrapper->_grad);
            this->replaceTensor(tensor_wrapper->_grad, std::move(grad));
        }

    });
    // mgb_log("Attach Tensor %d", id);
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

void EngineWrapper::backward(int32_t id){
    auto tensor = getTensor(id);
    auto dy = interpreter_for_js->put(makeGrad(tensor->shape()), true);
    Tensor* pdy = new Tensor(dy);
    std::vector<Tensor*> ts{tensor.get()};
    std::vector<Tensor*> gs{pdy};
    _engine.backward(ts, gs);
    // mgb_log("Backward Finish");
}


#ifdef __EMSCRIPTEN__
int EngineWrapper::registerTensorEM(const emscripten::val &v, const emscripten::val &data, const int type = 0){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(type));
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

    /*
    for(size_t i = 0; i < shape.total_nr_elems(); i++){
        ptr[i] = rdata[i];
    }
    */

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = _engine.registerTensor(tensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::randn(const emscripten::val &v, const float mean, const float std){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val memoryView{emscripten::typed_memory_view(l, rv.data())};
    memoryView.call<void>("set", v);
    TensorShape shape = TensorShape{l};
    auto op = GaussianRNG::make(rand(), mean, std);
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (uint32_t i=0; i<l; i++) {
        ptr[i] = rv[i];
        // mgb_log("size: %d", rv[i]);
    }
    auto handle = interpreter_for_js->put(*ret, true);
    // shape tensor
    auto tensor = std::make_shared<Tensor>(handle);

    // generate rand Tensor
    auto result = js::apply(op, tensor.get())[0];
    auto id = registerTensor(result);
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::zeros(const emscripten::val &v, int data_type = 0){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);
    TensorShape shape = TensorShape(rv);

    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(data_type));
    if(data_type == 0){
        auto ptr = ret->ptr<float>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0.0;
        }
    }
    else if(data_type == 1){
        auto ptr = ret->ptr<int32_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else if(data_type == 2){
        auto ptr = ret->ptr<int8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else if(data_type == 3){
        auto ptr = ret->ptr<uint8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else{
        throw std::runtime_error("Unknown type");
    }


    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = _engine.registerTensor(tensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::ones(const emscripten::val &v, int data_type = 0){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(data_type));
    if(data_type == 0){
        auto ptr = ret->ptr<float>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0.0;
        }
    }
    else if(data_type == 1){
        auto ptr = ret->ptr<int32_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else if(data_type == 2){
        auto ptr = ret->ptr<int8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else if(data_type == 3){
        auto ptr = ret->ptr<uint8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 0;
        } 
    }
    else{
        throw std::runtime_error("Unknown type");
    }

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = _engine.registerTensor(tensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::reshape(int a, const emscripten::val &v, int unspec){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val memoryView{emscripten::typed_memory_view(l, rv.data())};
    memoryView.call<void>("set", v);
    TensorShape shape = TensorShape{l};
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (uint32_t i=0; i<l; i++) {
        ptr[i] = rv[i];
    }
    auto handle = interpreter_for_js->put(*ret, true);
    // shape tensor
    auto shapeTensor = std::make_shared<Tensor>(handle);

    auto tensor = getTensor(a);
    auto op = Reshape::make(unspec != -1 ? unspec : 7);
    auto outTensor = js::apply(op, tensor.get(), shapeTensor.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
}

int EngineWrapper::removeAxis(int a, const emscripten::val &v){
    std::vector<int32_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    auto tensorA = getTensor(a);
    auto op = RemoveAxis::make(rv);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::addAxis(int a, const emscripten::val &v){
    std::vector<int32_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    auto tensorA = getTensor(a);
    auto op = AddAxis::make(rv);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

#endif




int EngineWrapper::index_one_hot(int a, int index, int axis){
    auto tensorA = getTensor(a);
    auto indexTensor = getTensor(index);
    auto op = IndexingOneHot::make(axis);
    auto outTensor = js::apply(op, tensorA.get(), indexTensor.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::registerTensor(std::shared_ptr<Tensor> t){
    auto id = _engine.registerTensor(t);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id;
}
int EngineWrapper::replaceTensor(int id, std::shared_ptr<Tensor> t){
    _engine.replaceTensor(id, t);
    return id;
}


int32_t EngineWrapper::getTensorOffset(const int id, int dtype){
    auto tensor = getTensor(id);
    if(dtype == 0){
        auto ptr = tensor->value().ptr<float>();
        #ifdef __EMSCRIPTEN__
        return reinterpret_cast<int32_t>(ptr);
        #else 
        return reinterpret_cast<uintptr_t>(ptr);
        #endif
    }
    else if(dtype == 1){
        auto ptr = tensor->value().ptr<int32_t>();
        #ifdef __EMSCRIPTEN__
        return reinterpret_cast<int32_t>(ptr);
        #else 
        return reinterpret_cast<uintptr_t>(ptr);
        #endif 
    }
    else if(dtype == 2){
        auto ptr = tensor->value().ptr<int8_t>();
        #ifdef __EMSCRIPTEN__
        return reinterpret_cast<int32_t>(ptr);
        #else 
        return reinterpret_cast<uintptr_t>(ptr);
        #endif 
    }
    else if(dtype == 3){
        auto ptr = tensor->value().ptr<uint8_t>();
        #ifdef __EMSCRIPTEN__
        return reinterpret_cast<int32_t>(ptr);
        #else 
        return reinterpret_cast<uintptr_t>(ptr);
        #endif 
    }
    else{
        throw std::runtime_error("Unknown type");
    }

}

int32_t EngineWrapper::getGradOffset(const int id, int dtype){
    auto gradID = getGradID(id);
    return getTensorOffset(gradID, dtype);
}

std::string EngineWrapper::getTensorShape(const int id){
    auto tensor = getTensor(id);
    return tensor->shape().to_string();
}

void printTensor(std::shared_ptr<Tensor> t){
    auto t_out_value = t->value().ptr<float>();
    for(size_t i = 0; i < t->shape().total_nr_elems(); i++){
        std::cout << t_out_value[i] << std::endl;
    }
}

void EngineWrapper::printTensor(int id){
    auto tensor = getTensor(id);
    auto ptr = tensor->value().ptr<float>();
    mgb_log("Tensor %d", id);
    for(size_t i = 0; i < tensor->shape().total_nr_elems(); i++){
        mgb_log("Tensor<%zu>: %f",i, ptr[i]);
    }
}

void EngineWrapper::printGrad(int id){
    auto grad = getGrad(id);
    auto ptr = grad->value().ptr<float>();
    mgb_log("Grad %d", id);
    for(size_t i = 0; i < grad->shape().total_nr_elems(); i++){
        mgb_log("Grad<%zu>: %f",i, ptr[i]);
    }
}

int EngineWrapper::mul(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::MUL);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::matmul(int a, int b, bool transposeA = false, bool transposeB = false){
    auto op = MatrixMul::make(
        transposeA, transposeB, 
        MatrixMul::ComputeMode::DEFAULT, MatrixMul::Format::DEFAULT, 
        MatrixMul::Strategy::HEURISTIC, 18446744073709551615ull);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::add_(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::ADD);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = replaceTensor(a, outTensor);
    return id;
}

int EngineWrapper::sub_(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::SUB);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = replaceTensor(a, outTensor);
    return id;
}

int EngineWrapper::add(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::ADD);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::sub(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::SUB);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::div(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::TRUE_DIV);
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::sin(int a){
    auto op = Elemwise::make(Elemwise::Mode::SIN);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
}


int EngineWrapper::cos(int a){
    auto op = Elemwise::make(Elemwise::Mode::COS);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
}

int EngineWrapper::log(int a){
    auto op = Elemwise::make(Elemwise::Mode::LOG);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
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



/*
SUM = 0,
//! sum of x * x for each element x
SUM_SQR = 1,
PRODUCT = 2,
MIN = 3,
MAX = 4,
MEAN = 5
*/

int EngineWrapper::reduce(int a, int mode, int axis){
    auto tensorA = getTensor(a);
    auto op = Reduce::make(static_cast<Reduce::Mode>(mode), axis, Reduce::DataType::DEFAULT);
    auto outTensor = js::apply(op, tensorA)[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::conv2d(int a, int w, const int stride, const int padding){
    auto tensorA = getTensor(a);
    auto weight = getTensor(w);
    auto op = Convolution::make(
        Convolution::Mode::CROSS_CORRELATION, padding, padding, stride, stride, 1, 1, 
        Convolution::Sparse::DENSE, Convolution::Format::NCHW, Convolution::ComputeMode::DEFAULT, Convolution::Strategy::HEURISTIC, 18446744073709551615ull);
    auto outTensor = js::apply(op, tensorA.get(), weight.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::pool(int a, const int kernel, const int stride, const int padding, const int mode){
    auto tensorA = getTensor(a);
    auto op = Pooling::make(
        static_cast<Pooling::Mode>(mode), padding, padding, stride, stride, kernel, kernel, Pooling::Format::NCHW); 
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::relu(int a){
    auto op = Elemwise::make(Elemwise::Mode::RELU);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
}


int EngineWrapper::exp(int a){
    auto op = Elemwise::make(Elemwise::Mode::EXP);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id; 
}

int EngineWrapper::typeCvt(int a, int type){
    auto op = TypeCvt::make(getDataType(type));
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::argmax(int a, int axis){
    auto op = Argmax::make(axis);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}



#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(Engine) {
  emscripten::class_<EngineWrapper>("Engine")
    .constructor<>()
    .function("startScope", &EngineWrapper::startScope)
    .function("endScope", &EngineWrapper::endScope)
    .function("attach", &EngineWrapper::attach)
    .function("backward", &EngineWrapper::backward)
    .function("registerTensor", &EngineWrapper::registerTensorEM)
    .function("zeros", &EngineWrapper::zeros)
    .function("ones", &EngineWrapper::ones)
    .function("disposeTensor", &EngineWrapper::disposeTensor)
    .function("getTensorOffset", &EngineWrapper::getTensorOffset)
    .function("getGradOffset", &EngineWrapper::getGradOffset)
    .function("getGrad", &EngineWrapper::getGradID)
    .function("randn", &EngineWrapper::randn)
    .function("mul", &EngineWrapper::mul)
    .function("div", &EngineWrapper::div)
    .function("matmul", &EngineWrapper::matmul)
    .function("add", &EngineWrapper::add)
    .function("sub", &EngineWrapper::sub)
    .function("add_", &EngineWrapper::add_)
    .function("sub_", &EngineWrapper::sub_)
    .function("sin", &EngineWrapper::sin)
    .function("cos", &EngineWrapper::cos)
    .function("conv2d", &EngineWrapper::conv2d)
    .function("pool", &EngineWrapper::pool)
    .function("relu", &EngineWrapper::relu)
    .function("reshape", &EngineWrapper::reshape)
    .function("log", &EngineWrapper::log)
    .function("reduce", &EngineWrapper::reduce)
    .function("removeAxis", &EngineWrapper::removeAxis)
    .function("addAxis", &EngineWrapper::addAxis)
    .function("index_one_hot", &EngineWrapper::index_one_hot)
    .function("exp", &EngineWrapper::exp)
    .function("getTensorShape", &EngineWrapper::getTensorShape)
    .function("astype", &EngineWrapper::typeCvt)
    .function("argmax", &EngineWrapper::argmax)
    ;

}
#endif

}