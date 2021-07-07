#include "./webassembly.h"
#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"
#include <vector>
#include <unordered_map>

using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {

class Engine{

public:
    Engine();
    ~Engine();

    static Engine& inst();

    void startScope();
    void endScope();
    void attach(Tensor* t, CallbackFunc&& callback);
    void backward(std::vector<Tensor*> tensors, std::vector<Tensor*> grads);

    void insertTensor(int id, std::shared_ptr<TensorWrapper> tensor){
        tensor_registry->insert({id, tensor});
    }

    std::shared_ptr<TensorWrapper> getTensor(int id){
        return tensor_registry->at(id);
    }

private:
    std::shared_ptr<GradKey> gradkey;
    std::shared_ptr<std::unordered_map<int, std::shared_ptr<TensorWrapper>>> tensor_registry;
    bool inScope;
};

class EngineWrapper{
public:
    EngineWrapper(){}

    void startScope(){
        _engine.startScope();
    }
    void endScope(){
        _engine.endScope();
    }
    void attach(int32_t id);
    int backward(int32_t id);

    #ifdef __EMSCRIPTEN__
    int registerTensor(const int id, const emscripten::val &v, const emscripten::val &data);
    int randn(const int id, const emscripten::val &v);
    #else
    int registerTensor(std::shared_ptr<Tensor> t);
    #endif
    int32_t getTensorOffset(const int id);
    int32_t getGradOffset(const int id);
    std::shared_ptr<TensorWrapper> getTensor(int id){
        return _engine.getTensor(id);
    }

    int mul(int a, int b);
    int add(int a, int b);
    int sub(int a, int b);
    int sin(int a);
    int cos(int a);

    int nextTensorID = 2;

private:
    Engine _engine;
};


};