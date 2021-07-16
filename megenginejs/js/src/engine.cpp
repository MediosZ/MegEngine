#include "./engine.h"
#include "./tensor.h"

namespace mgb::imperative::js {

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
    auto tensor = getTensorWrapper(id);
    _engine.disposeTensor(tensor->_tensor);
    _engine.disposeTensor(tensor->_grad);
    _tensor_wrapper_registry->erase(id);
}

void EngineWrapper::attach(int32_t id){
    auto tensor_wrapper = getTensorWrapper(id);
    auto tensor = getTensor(id);
    _engine.attach(tensor.get(), [tensor_wrapper, this](std::shared_ptr<Tensor> grad){
        auto id = _engine.registerTensor(std::move(grad));
        tensor_wrapper->_grad = id;
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
int EngineWrapper::registerTensorEM(const emscripten::val &v, const emscripten::val &data){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);

    SmallVector<float> rdata;
    const auto ldata = data["length"].as<unsigned>();
    rdata.resize(ldata);
    emscripten::val dataMemoryView{emscripten::typed_memory_view(ldata, rdata.data())};
    dataMemoryView.call<void>("set", data);

    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    auto ptr = ret->ptr<float>();
    for(size_t i = 0; i < shape.total_nr_elems(); i++){
        ptr[i] = rdata[i];
    }

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

int EngineWrapper::zeros(const emscripten::val &v){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    auto ptr = ret->ptr<float>();
    for(size_t i = 0; i < shape.total_nr_elems(); i++){
        ptr[i] = 0.0;
    }

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = _engine.registerTensor(tensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::ones(const emscripten::val &v){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val shapeMemoryView{emscripten::typed_memory_view(l, rv.data())};
    shapeMemoryView.call<void>("set", v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape);
    auto ptr = ret->ptr<float>();
    for(size_t i = 0; i < shape.total_nr_elems(); i++){
        ptr[i] = 1.0;
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

#endif

int EngineWrapper::registerTensor(std::shared_ptr<Tensor> t){
    auto id = _engine.registerTensor(t);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id;
}
int EngineWrapper::replaceTensor(int id, std::shared_ptr<Tensor> t){
    _engine.replaceTensor(id, t);
    return id;
}


int32_t EngineWrapper::getTensorOffset(const int id){
    auto tensor = getTensor(id);
    auto ptr = tensor->value().ptr<float>();
    #ifdef __EMSCRIPTEN__
    return reinterpret_cast<int32_t>(ptr);
    #else 
    return reinterpret_cast<uintptr_t>(ptr);
    #endif
}

int32_t EngineWrapper::getGradOffset(const int id){
    auto gradID = getGradID(id);
    return getTensorOffset(gradID);
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

int EngineWrapper::mean(int a){
    auto op = Reduce::make(Reduce::Mode::MEAN);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get(), make_shape({1}).get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

int EngineWrapper::min(int a){
    auto op = Reduce::make(Reduce::Mode::MIN);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get(), make_shape({1}).get())[0];
    auto id = registerTensor(outTensor);
    return id;
}
int EngineWrapper::max(int a){
    auto op = Reduce::make(Reduce::Mode::MAX);
    auto tensorA = getTensor(a);
    auto outTensor = js::apply(op, tensorA.get(), make_shape({1}).get())[0];
    auto id = registerTensor(outTensor);
    return id;
}

std::shared_ptr<Tensor> flatten(std::shared_ptr<Tensor> t){
    auto reshape = Reshape::make(0);
    return js::apply(reshape, t.get(), make_shape({-1}))[0];
}


int EngineWrapper::sum(int a){
    auto tensorA = getTensor(a);
    auto op = Reduce::make(Reduce::Mode::SUM, 0, Reduce::DataType::DEFAULT);
    auto outTensor = js::apply(op, flatten(tensorA).get())[0];
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


std::vector<int> returnVec(){
    std::vector<int> vec{1,2,3};
    return vec;
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
    .function("mean", &EngineWrapper::mean)
    .function("min", &EngineWrapper::min)
    .function("max", &EngineWrapper::max)
    .function("sum", &EngineWrapper::sum)
    .function("conv2d", &EngineWrapper::conv2d)
    .function("pool", &EngineWrapper::pool)
    .function("relu", &EngineWrapper::relu)
    .function("reshape", &EngineWrapper::reshape)
    
    .function("getTensorShape", &EngineWrapper::getTensorShape)
    ;

    emscripten::function("returnVec", &returnVec);

    emscripten::register_vector<int>("vector<int>");
}
#endif

}