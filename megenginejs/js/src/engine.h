#include "./webassembly.h"
#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"
#include "megbrain/imperative/ops/rng.h"
#include <vector>
#include <unordered_map>
#include <map>
#include "./utils.h"

using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {
class EngineWrapper{
public:
    EngineWrapper();
    ~EngineWrapper();
    void startScope();
    void endScope();
    void _attach(Tensor* t, CallbackFunc&& callback);
    void _backward(std::vector<Tensor*> tensors, std::vector<Tensor*> grads);
    void attach(int32_t id);
    void backward(int32_t id);

    int apply(std::shared_ptr<OpDef> op, int a);
    int apply(std::shared_ptr<OpDef> op, int a, int b);

    #ifdef __EMSCRIPTEN__
    int replaceTensorWithDataEM(int a, const emscripten::val &v, const emscripten::val &data, const int type);
    int replaceTensorWithIDEM(int a, const int new_id);
    int registerTensorEM(const emscripten::val &v, const emscripten::val &data, const int type);
    #endif

    int _registerTensor(std::shared_ptr<Tensor> tensor);
    int registerTensor(std::shared_ptr<Tensor> t);
    int replaceTensor(int id, std::shared_ptr<Tensor> t);
    int replaceTensor(int id, int new_id);

    #ifdef __EMSCRIPTEN__
    using TensorOffset = int32_t;
    #else 
    using TensorOffset = uintptr_t;
    #endif
    TensorOffset getTensorOffset(const int id, int dtype);
    TensorOffset getGradOffset(const int id, int dtype);
    std::shared_ptr<Tensor> getTensor(int id){
        // mgb_log("about to get tensor %d, but size is %d", id, _tensor_registry.size());
        return _tensor_registry.at(id);
    }
    std::shared_ptr<Tensor> getGrad(int id){
        auto gradID = _tensor_wrapper_registry.at(id)->_grad;
        return getTensor(gradID);
    }
    int getGradID(int id){
        return _tensor_wrapper_registry.at(id)->_grad;
    }
    std::string getTensorShape(const int id);
    std::shared_ptr<TensorWrapper> getTensorWrapper(int id){
        return _tensor_wrapper_registry.at(id);
    }
    size_t size(){
        return _tensor_wrapper_registry.size();
    }
    void printTensor(int id);
    void printGrad(int id);
    void disposeTensor(int id);

private:
    std::map<int, std::shared_ptr<TensorWrapper>> _tensor_wrapper_registry;
    // originally in Engine
    std::shared_ptr<GradKey> gradkey;
    std::map<int, std::shared_ptr<Tensor>> _tensor_registry;
    bool inScope;
    int nextTensorID;
};

static EngineWrapper* _inst;
EngineWrapper* EngineWrapperInst();


#ifdef __EMSCRIPTEN__
int replaceTensorWithDataEM(int a, const emscripten::val &v, const emscripten::val &data, const int type);
int replaceTensorWithIDEM(int a, const int new_id);
int registerTensorEM(const emscripten::val &v, const emscripten::val &data, const int type);
int randn(const emscripten::val &v, const float mean, const float std);
int ones(const emscripten::val &v, int data_type);
int zeros(const emscripten::val &v, int data_type);
int reshape(int a, const emscripten::val &v, int unspec);
int broadcast_to(int a, const emscripten::val &v);
int removeAxis(int a, const emscripten::val &v);
int addAxis(int a, const emscripten::val &v);
#endif
int mul(int a, int b);
int div(int a, int b);
int matmul(int a, int b, bool transposeA, bool transposeB);
int batch_matmul(int a, int b, bool transposeA, bool transposeB);
int add(int a, int b);
int sub(int a, int b);
int add_(int a, int b);
int sub_(int a, int b);
int sin(int a);
int cos(int a);
int conv2d(int a, int weight, const int stride, const int padding);
int pool(int a, const int kernel, const int stride, const int padding, const int mode);
int relu(int a);
int log(int a);
int reduce(int a, int mode, int axis);
int exp(int a);
int dot(int a, int b);
int typeCvt(int a, int type);
int argmax(int a, int axis);
int index_one_hot(int a, int index, int axis);
int eq(int a, int b);


};