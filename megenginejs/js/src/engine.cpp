#include "./engine.h"
#include "./tensor.h"

namespace mgb::imperative::js {

EngineWrapper::EngineWrapper() {
    inScope = true;
    nextTensorID = 0;
    // mgb_log("creating EngineWrapper");
}
EngineWrapper::~EngineWrapper(){
    // mgb_log("Delete Engine, release %lu tensors", _tensor_registry.size());
}

int EngineWrapper::apply(std::shared_ptr<OpDef> op, int a, int b){
    auto tensorA = getTensor(a);
    auto tensorB = getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::apply(std::shared_ptr<OpDef> op, int a){
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = registerTensor(outTensor);
    return id;
}


//! this function should not be marked as static
EngineWrapper* EngineWrapperInst(){
    if(_inst == NULL){
        _inst = new EngineWrapper;
    }
    // mgb_log("wrapper: %p", _inst);
    return _inst;
}

void EngineWrapper::startScope(){
    gradkey = std::make_shared<GradKey>();
    inScope = true;
}

void EngineWrapper::endScope(){
    inScope = false;
}

void EngineWrapper::_attach(Tensor* t, CallbackFunc&& callback){
    mgb_assert(inScope, "attach must be called inside a scope");
    gradkey->attach(t, std::move(callback));
}

void EngineWrapper::_backward(std::vector<Tensor*> tensors, std::vector<Tensor*> grads){
    mgb_assert(inScope, "backward must be called inside a scope");
    gradkey->backward(tensors, grads);
}

int EngineWrapper::_registerTensor(std::shared_ptr<Tensor> tensor){
    auto id = nextTensorID++;
    _tensor_registry.insert(std::make_pair(id, tensor));
    return id;
}

void EngineWrapper::disposeTensor(int id){
    
    auto tensor = getTensorWrapper(id);
    // mgb_log("disposeTensor %d %d", id, tensor->_grad);
    _tensor_registry.erase(tensor->_tensor);
    if(tensor->_grad != -1){
        _tensor_registry.erase(tensor->_grad);
    }
    _tensor_wrapper_registry.erase(id);
}

void EngineWrapper::attach(int32_t id){
    auto tensor_wrapper = getTensorWrapper(id);
    auto tensor = getTensor(id);
    // mgb_log("attach %d", id);
    _attach(tensor.get(), [tensor_wrapper, this](std::shared_ptr<Tensor> grad){
        // there is no grad in current tensor
        // mgb_log("receive tensor: %d, grad: %d", tensor_wrapper->_tensor, tensor_wrapper->_grad);
        if(tensor_wrapper->_grad == -1){
            auto id = _registerTensor(std::move(grad));
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

void EngineWrapper::backward(int32_t id){
    auto tensor = getTensor(id);
    auto dy = interpreter_for_js->put(makeGrad(tensor->shape()), true);
    Tensor* pdy = new Tensor(dy);
    std::vector<Tensor*> ts{tensor.get()};
    std::vector<Tensor*> gs{pdy};
    _backward(ts, gs);
}


#ifdef __EMSCRIPTEN__

int EngineWrapper::replaceTensorWithDataEM(int a, const emscripten::val &v, const emscripten::val &data, const int type){
    auto rv = getVectorFromVal(v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu:default");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(type));
    assignData(data, ret, type);

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    replaceTensor(a, tensor);
    return a;
}
int EngineWrapper::replaceTensorWithIDEM(int a, const int new_id){
    replaceTensor(a, new_id);
    return a;
}

int EngineWrapper::registerTensorEM(const emscripten::val &v, const emscripten::val &data, const int type = 0){
    auto rv = getVectorFromVal(v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu:default");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(type));
    assignData(data, ret, type);
    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = registerTensor(tensor);
    // mgb_log("register Tensor %d", id);
    return id;
}

int randn(const emscripten::val &v, const float mean, const float std){
    auto rv = getVectorFromVal(v);
    const auto l = v["length"].as<unsigned>();
    TensorShape shape = TensorShape{l};
    auto seed = rand();
    auto cn = CompNode::load("cpu:default");
    auto rngHandle = rng::new_handle(cn, seed);
    auto op = GaussianRNG::make(seed, mean, std, dtype::Float32(), rngHandle);
    
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
    auto wrapper = EngineWrapperInst();
    auto id = wrapper->registerTensor(result);
    // mgb_log("register Tensor %d", id);
    return id;
}

int zeros(const emscripten::val &v, int data_type = 0){
    auto rv = getVectorFromVal(v);
    TensorShape shape = TensorShape(rv);

    auto cn = CompNode::load("cpu:default");
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
    auto wrapper = EngineWrapperInst();
    auto id = wrapper->registerTensor(tensor);
    // mgb_log("register Tensor %d", id);
    return id;
}

int ones(const emscripten::val &v, int data_type = 0){
    auto rv = getVectorFromVal(v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu:default");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(data_type));
    if(data_type == 0){
        auto ptr = ret->ptr<float>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 1.0;
        }
    }
    else if(data_type == 1){
        auto ptr = ret->ptr<int32_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 1;
        } 
    }
    else if(data_type == 2){
        auto ptr = ret->ptr<int8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 1;
        } 
    }
    else if(data_type == 3){
        auto ptr = ret->ptr<uint8_t>();
        for(size_t i = 0; i < shape.total_nr_elems(); i++){
            ptr[i] = 1;
        } 
    }
    else{
        throw std::runtime_error("Unknown type");
    }

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto wrapper = EngineWrapperInst();
    auto id = wrapper->registerTensor(tensor);
    // mgb_log("register Tensor %d", id);
    return id;
}

int reshape(int a, const emscripten::val &v, int unspec){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val memoryView{emscripten::typed_memory_view(l, rv.data())};
    memoryView.call<void>("set", v);
    TensorShape shape = TensorShape{l};
    auto cn = CompNode::load("cpu:default");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (uint32_t i=0; i<l; i++) {
        ptr[i] = rv[i];
    }
    auto handle = interpreter_for_js->put(*ret, true);
    // shape tensor
    auto shapeTensor = std::make_shared<Tensor>(handle);
    auto op = Reshape::make(unspec != -1 ? unspec : 7);

    auto wrapper = EngineWrapperInst();
    auto tensor = wrapper->getTensor(a);
    auto outTensor = js::apply(op, tensor.get(), shapeTensor.get())[0];
    auto id = wrapper->registerTensor(outTensor);
    return id; 
}


int broadcast_to(int a, const emscripten::val &v){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val memoryView{emscripten::typed_memory_view(l, rv.data())};
    memoryView.call<void>("set", v);
    TensorShape shape = TensorShape{l};
    auto cn = CompNode::load("cpu:default");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (uint32_t i=0; i<l; i++) {
        ptr[i] = rv[i];
    }
    auto handle = interpreter_for_js->put(*ret, true);
    // shape tensor
    auto shapeTensor = std::make_shared<Tensor>(handle);
    auto op = Broadcast::make();
    
    auto wrapper = EngineWrapperInst();
    auto tensor = wrapper->getTensor(a);
    auto outTensor = js::apply(op, tensor.get(), shapeTensor.get())[0];
    auto id = wrapper->registerTensor(outTensor);
    return id;
}

int removeAxis(int a, const emscripten::val &v){
    std::vector<int32_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    auto op = RemoveAxis::make(rv);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
}

int addAxis(int a, const emscripten::val &v){
    std::vector<int32_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    auto op = AddAxis::make(rv);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
}

#endif

int EngineWrapper::registerTensor(std::shared_ptr<Tensor> t){
    auto id = _registerTensor(t);
    auto pair = std::make_pair(id, std::make_shared<TensorWrapper>(id));
    _tensor_wrapper_registry.insert(pair);
    return id;
}

int EngineWrapper::replaceTensor(int id, std::shared_ptr<Tensor> t){
    _tensor_registry.at(id) = t;
    return id;
}

int EngineWrapper::replaceTensor(int id, int new_id){
    _tensor_registry.at(id) = getTensor(new_id);
    disposeTensor(new_id);
    return id;
}


EngineWrapper::TensorOffset EngineWrapper::getTensorOffset(const int id, int dtype){
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


EngineWrapper::TensorOffset EngineWrapper::getGradOffset(const int id, int dtype){
    auto gradID = getGradID(id);
    return getTensorOffset(gradID, dtype);
}

std::string EngineWrapper::getTensorShape(const int id){
    auto tensor = getTensor(id);
    return tensor->shape().to_string();
}

void EngineWrapper::printTensor(int id){
    auto tensor = getTensor(id);
    mgb_log("print Tensor %d", id);
    auto ptr = tensor->value().ptr<float>();
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


int index_one_hot(int a, int index, int axis){
    auto op = IndexingOneHot::make(axis);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, index);
}

int mul(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::MUL);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}


int eq(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::EQ);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}


int matmul(int a, int b, bool transposeA = false, bool transposeB = false){
    auto op = MatrixMul::make(
        transposeA, transposeB, 
        MatrixMul::ComputeMode::DEFAULT, MatrixMul::Format::DEFAULT, 
        MatrixMul::Strategy::HEURISTIC, 18446744073709551615ull);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}

int batch_matmul(int a, int b, bool transposeA = false, bool transposeB = false){
    auto op = BatchedMatrixMul::make(
        transposeA, transposeB, 
        BatchedMatrixMul::ComputeMode::DEFAULT, BatchedMatrixMul::Format::DEFAULT, 
        BatchedMatrixMul::Strategy::HEURISTIC, 18446744073709551615ull);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}


int add_(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::ADD);
    auto wrapper = EngineWrapperInst();
    auto tensorA = wrapper->getTensor(a);
    auto tensorB = wrapper->getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = wrapper->replaceTensor(a, outTensor);
    return id;
}

int sub_(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::SUB);
    auto wrapper = EngineWrapperInst();
    auto tensorA = wrapper->getTensor(a);
    auto tensorB = wrapper->getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = wrapper->replaceTensor(a, outTensor);
    return id;
}

int add(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::ADD);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}

int sub(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::SUB);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}

int div(int a, int b){
    auto op = Elemwise::make(Elemwise::Mode::TRUE_DIV);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}

int sin(int a){
    auto op = Elemwise::make(Elemwise::Mode::SIN);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
}


int cos(int a){
    auto op = Elemwise::make(Elemwise::Mode::COS);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
}

int log(int a){
    auto op = Elemwise::make(Elemwise::Mode::LOG);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
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

int reduce(int a, int mode, int axis){
    auto op = Reduce::make(static_cast<Reduce::Mode>(mode), axis, Reduce::DataType::DEFAULT);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);
}

int conv2d(int a, int w, const int stride, const int padding){
    auto op = Convolution::make(
        Convolution::Mode::CROSS_CORRELATION, padding, padding, stride, stride, 1, 1, 
        Convolution::Sparse::DENSE, Convolution::Format::NCHW, Convolution::ComputeMode::DEFAULT, Convolution::Strategy::HEURISTIC, 18446744073709551615ull);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, w);
}

int pool(int a, const int kernel, const int stride, const int padding, const int mode){
    auto op = Pooling::make(
        static_cast<Pooling::Mode>(mode), padding, padding, stride, stride, kernel, kernel, Pooling::Format::NCHW); 
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a);;
}

int relu(int a){
    auto op = Elemwise::make(Elemwise::Mode::RELU);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a); 
}


int exp(int a){
    auto op = Elemwise::make(Elemwise::Mode::EXP);
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a); 
}

int dot(const int a, const int b){
    auto op = Dot::make();
    auto wrapper = EngineWrapperInst();
    return wrapper->apply(op, a, b);
}

int typeCvt(int a, int type){
    auto wrapper = EngineWrapperInst();
    auto op = TypeCvt::make(getDataType(type));
    return wrapper->apply(op, a);
}

int argmax(int a, int axis){
    auto wrapper = EngineWrapperInst();
    auto op = Argmax::make(axis);
    return wrapper->apply(op, a);
}


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(Engine) {
    emscripten::function("runBackward", &testJSBack);
    emscripten::function("initTensor", &initTensor);
    emscripten::function("inst", &EngineWrapperInst, emscripten::allow_raw_pointers());

emscripten::class_<EngineWrapper>("Engine")
    // .constructor<>()
    .smart_ptr<std::shared_ptr<EngineWrapper>>("Engine")
    //.class_function("inst", &EngineWrapper::Inst, emscripten::allow_raw_pointers())
    .function("startScope", &EngineWrapper::startScope)
    .function("endScope", &EngineWrapper::endScope)
    .function("attach", &EngineWrapper::attach)
    .function("backward", &EngineWrapper::backward)
    .function("registerTensor", &EngineWrapper::registerTensorEM)
    .function("replaceTensorWithData", &EngineWrapper::replaceTensorWithDataEM)
    .function("replaceTensorWithID", &EngineWrapper::replaceTensorWithIDEM)
    .function("disposeTensor", &EngineWrapper::disposeTensor)
    .function("getTensorOffset", &EngineWrapper::getTensorOffset)
    .function("getGradOffset", &EngineWrapper::getGradOffset)
    .function("getGrad", &EngineWrapper::getGradID)
    .function("size", &EngineWrapper::size)
    .function("getTensorShape", &EngineWrapper::getTensorShape)
    ;
    emscripten::function("randn", &randn);
    emscripten::function("zeros", &zeros);
    emscripten::function("ones", &ones);
    emscripten::function("eq", &eq);
    emscripten::function("mul", &mul);
    emscripten::function("div", &div);
    emscripten::function("matmul", &matmul);
    emscripten::function("batch_matmul", &batch_matmul);
    emscripten::function("add", &add);
    emscripten::function("sub", &sub);
    emscripten::function("add_", &add_);
    emscripten::function("sub_", &sub_);
    emscripten::function("sin", &sin);
    emscripten::function("cos", &cos);
    emscripten::function("conv2d", &conv2d);
    emscripten::function("pool", &pool);
    emscripten::function("relu", &relu);
    emscripten::function("reshape", &reshape);
    emscripten::function("broadcast_to", &broadcast_to);
    emscripten::function("log", &log);
    emscripten::function("reduce", &reduce);
    emscripten::function("removeAxis", &removeAxis);
    emscripten::function("addAxis", &addAxis);
    emscripten::function("index_one_hot", &index_one_hot);
    emscripten::function("exp", &exp);
    emscripten::function("astype", &typeCvt);
    emscripten::function("argmax", &argmax);
    emscripten::function("dot", &dot);

}
#endif

}
