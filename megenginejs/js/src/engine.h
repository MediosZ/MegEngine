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

    void insertTensor(int id, std::shared_ptr<Tensor> tensor){
        tensor_registry->insert({id, tensor});
    }

    int registerTensor(std::shared_ptr<Tensor> tensor);

    std::shared_ptr<Tensor> getTensor(int id){
        return tensor_registry->at(id);
    }

    void disposeTensor(int id){
        tensor_registry->erase(id);
    }

private:
    std::shared_ptr<GradKey> gradkey;
    std::shared_ptr<std::unordered_map<int, std::shared_ptr<Tensor>>> tensor_registry;
    bool inScope;
    int nextTensorID = 0;

};

class EngineWrapper{
public:
    EngineWrapper();

    void startScope(){
        _engine.startScope();
    }
    void endScope(){
        _engine.endScope();
    }
    void attach(int32_t id);
    void backward(int32_t id);

    #ifdef __EMSCRIPTEN__
    int registerTensorEM(const emscripten::val &v, const emscripten::val &data);
    int randn(const emscripten::val &v, const float mean, const float std);
    #endif
    int registerTensor(std::shared_ptr<Tensor> t);

    int32_t getTensorOffset(const int id);
    int32_t getGradOffset(const int id);
    std::shared_ptr<Tensor> getTensor(int id){
        return _engine.getTensor(id);
    }
    std::shared_ptr<Tensor> getGrad(int id){
        auto gradID = _tensor_wrapper_registry->at(id)->_grad;
        return getTensor(gradID);
    }
    int getGradID(int id){
        return _tensor_wrapper_registry->at(id)->_grad;
    }
    std::shared_ptr<TensorWrapper> getTensorWrapper(int id){
        return _tensor_wrapper_registry->at(id);
    }

    void printTensor(int id);
    void printGrad(int id);

    void disposeTensor(int id);

    int mul(int a, int b);
    int div(int a, int b);
    int matmul(int a, int b);
    int add(int a, int b);
    int sub(int a, int b);
    int sin(int a);
    int cos(int a);
    int mean(int a);
    int max(int a);
    int min(int a);
    int sum(int a);
private:
    Engine _engine;
    std::shared_ptr<std::unordered_map<int, std::shared_ptr<TensorWrapper>>> _tensor_wrapper_registry;
};


};