#include "./engine.h"
#include "./tensor.h"

namespace mgb::imperative::js {

Engine& Engine::inst() {
    static Engine inst_;
    return inst_;
}

Engine::Engine(){
    mgb_log("Create Engine");
    inScope = true;
    #ifndef __EMSCRIPTEN__
    tensor_registry = std::make_shared<std::unordered_map<int, std::shared_ptr<Tensor>>>();
    #endif
}

Engine::~Engine(){

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
    #ifndef __EMSCRIPTEN__
    _tensor_wrapper_registry = std::make_shared<std::unordered_map<int, std::shared_ptr<TensorWrapper>>>();
    #endif
}

void EngineWrapper::attach(int32_t id){
    auto tensor_wrapper = getTensorWrapper(id);
    auto tensor = _engine.getTensor(id);
    // std::cout << tensor << std::endl;
    _engine.attach(tensor.get(), [tensor_wrapper, this](std::shared_ptr<Tensor> grad){
        // std::cout << tensor << std::endl;
        auto id = _engine.registerTensor(std::move(grad));
        tensor_wrapper->_grad = id;
/*         std::cout <<tensor->_grad.get() << std::endl;
        HostTensorND gradTensor = tensor->_grad->value();
        auto t_out_grad = gradTensor.ptr<float>();
        for(int i = 0; i < tensor->_grad->shape().total_nr_elems(); i++){
            // std::cout << t_out_grad[i] << std::endl;
            mgb_log("Grad<%d>: %f",i, t_out_grad[i]);
        } */
    });
    mgb_log("Attach Tensor %d", id);
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
    auto tensor = _engine.getTensor(id);
    auto dy = interpreter_for_js->put(makeGrad(tensor->shape()), true);
    Tensor* pdy = new Tensor(dy);
    // auto dy = make_tensor_like(tensor->_tensor.get(), 1);
    // auto dy_id = nextTensorID++;
    // _engine.insertTensor(dy_id, std::make_shared<TensorWrapper>(dy));
    std::vector<Tensor*> ts{tensor.get()};
    std::vector<Tensor*> gs{pdy};
    _engine.backward(ts, gs);
    mgb_log("Backward Finish");
}


#ifdef __EMSCRIPTEN__
int EngineWrapper::registerTensor(const emscripten::val &v, const emscripten::val &data){
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
    for(int i = 0; i < shape.total_nr_elems(); i++){
        ptr[i] = rdata[i];
    }

    auto handle = interpreter_for_js->put(*ret, true);
    auto tensor = std::make_shared<Tensor>(handle);
    auto id = _engine.registerTensor(tensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}

int EngineWrapper::randn(const emscripten::val &v){
    SmallVector<size_t> rv;
    const auto l = v["length"].as<unsigned>();
    rv.resize(l);
    emscripten::val memoryView{emscripten::typed_memory_view(l, rv.data())};
    memoryView.call<void>("set", v);
    TensorShape shape = TensorShape{l};
    auto op = std::shared_ptr<OpDef>(GaussianRNG::make(rand()));
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, dtype::Int32());
    auto ptr = ret->ptr<int32_t>();
    for (int i=0; i<l; i++) {
        ptr[i] = rv[i];
        // mgb_log("size: %d", rv[i]);
    }
    auto handle = interpreter_for_js->put(*ret, true);
    // shape tensor
    auto tensor = std::make_shared<Tensor>(handle);

    // generate rand Tensor
    auto result = js::apply(op, tensor.get())[0];
    auto id = _engine.registerTensor(result);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    // mgb_log("register Tensor %d", id);
    return id;
}
#else 
int EngineWrapper::registerTensor(std::shared_ptr<Tensor> t){
    auto id = _engine.registerTensor(t);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id;
}
#endif

int32_t EngineWrapper::getTensorOffset(const int id){
    auto tensor = _engine.getTensor(id);
    auto ptr = tensor->value().ptr<float>();
    #ifdef __EMSCRIPTEN__
    return reinterpret_cast<int32_t>(ptr);
    #else 
    return reinterpret_cast<uintptr_t>(ptr);
    #endif
}

int32_t EngineWrapper::getGradOffset(const int id){
    auto tensor_wrapper = getTensorWrapper(id);
    auto grad = _engine.getTensor(tensor_wrapper->_grad);
    // std::cout << tensor->_grad.get() << " " << tensor->_tensor.get() << std::endl;
    auto ptr = grad->value().ptr<float>();
    #ifdef __EMSCRIPTEN__
    return reinterpret_cast<int32_t>(ptr);
    #else 
    return reinterpret_cast<uintptr_t>(ptr);
    #endif
}

void printTensor(std::shared_ptr<Tensor> t){
    auto t_out_value = t->value().ptr<float>();
    for(int i = 0; i < t->shape().total_nr_elems(); i++){
        std::cout << t_out_value[i] << std::endl;
    }
}

int EngineWrapper::mul(int a, int b){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::MUL));
    auto tensorA = _engine.getTensor(a);
    auto tensorB = _engine.getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = _engine.registerTensor(outTensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id;
}

int EngineWrapper::add(int a, int b){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto tensorA = _engine.getTensor(a);
    auto tensorB = _engine.getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = _engine.registerTensor(outTensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});;
    return id;
}

int EngineWrapper::sub(int a, int b){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::SUB));
    auto tensorA = _engine.getTensor(a);
    auto tensorB = _engine.getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = _engine.registerTensor(outTensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id;
}

int EngineWrapper::sin(int a){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::SIN));
    auto tensorA = _engine.getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = _engine.registerTensor(outTensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
    return id; 
}


int EngineWrapper::cos(int a){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::COS));
    auto tensorA = _engine.getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = _engine.registerTensor(outTensor);
    _tensor_wrapper_registry->insert({id, std::make_shared<TensorWrapper>(id)});
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
    .function("registerTensor", &EngineWrapper::registerTensor, emscripten::allow_raw_pointers())
    .function("getTensorOffset", &EngineWrapper::getTensorOffset)
    .function("getGradOffset", &EngineWrapper::getGradOffset)
    .function("randn", &EngineWrapper::randn)
    .function("mul", &EngineWrapper::mul)
    .function("add", &EngineWrapper::add)
    .function("sub", &EngineWrapper::sub)
    .function("sin", &EngineWrapper::sin)
    .function("cos", &EngineWrapper::cos)
    ;
}
#endif

}