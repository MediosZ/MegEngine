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

void Engine::attach(Tensor* t){
    mgb_assert(inScope, "attach must be called inside a scope");
    gradkey->attach(t);
}

void Engine::backward(std::vector<Tensor*> tensors, std::vector<Tensor*> grads){
    mgb_assert(inScope, "backward must be called inside a scope");
    gradkey->backward(tensors, grads);
}


void EngineWrapper::attach(int32_t id){
    auto tensor = _engine.getTensor(id);
    _engine.attach(tensor.get());
    mgb_log("Attach Tensor %d", id);
}

int EngineWrapper::backward(int32_t id){
    auto tensor = _engine.getTensor(id);;
    auto dy = make_tensor_like(tensor.get(), 1);
    auto dy_id = nextTensorID++;
    _engine.insertTensor(dy_id, dy);
    std::vector<Tensor*> ts{tensor.get()};
    std::vector<Tensor*> gs{dy.get()};
    _engine.backward(ts, gs);
    mgb_log("Backward Finish");
    return nextTensorID;
}


#ifdef __EMSCRIPTEN__
int EngineWrapper::registerTensor(const int id, const emscripten::val &v, const emscripten::val &data){
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
    _engine.insertTensor(id, tensor);
    // mgb_log("register Tensor %d", id);
    nextTensorID = id + 1;
    return id;
}

int EngineWrapper::randn(const int id, const emscripten::val &v){
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
    _engine.insertTensor(id, result);
    // mgb_log("register Tensor %d", id);
    nextTensorID = id + 1;
    return id;
}
#else 
int EngineWrapper::registerTensor(std::shared_ptr<Tensor> t){
    _engine.insertTensor(nextTensorID, t);
    return nextTensorID++;
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
    auto id = nextTensorID++;
    _engine.insertTensor(id, outTensor);
    return id;
}

int EngineWrapper::add(int a, int b){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::ADD));
    auto tensorA = _engine.getTensor(a);
    auto tensorB = _engine.getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = nextTensorID++;
    _engine.insertTensor(id, outTensor);
    return id;
}

int EngineWrapper::sub(int a, int b){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::SUB));
    auto tensorA = _engine.getTensor(a);
    auto tensorB = _engine.getTensor(b);
    auto outTensor = js::apply(op, tensorA.get(), tensorB.get())[0];
    auto id = nextTensorID++;
    _engine.insertTensor(id, outTensor);
    return id;
}

int EngineWrapper::sin(int a){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::SIN));
    auto tensorA = _engine.getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = nextTensorID++;
    _engine.insertTensor(id, outTensor);
    return id; 
}


int EngineWrapper::cos(int a){
    auto op = std::shared_ptr<OpDef>(Elemwise::make(Elemwise::Mode::COS));
    auto tensorA = _engine.getTensor(a);
    auto outTensor = js::apply(op, tensorA.get())[0];
    auto id = nextTensorID++;
    _engine.insertTensor(id, outTensor);
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