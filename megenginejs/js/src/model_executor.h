#include "./webassembly.h"
#include "./grad.h"
#include "megbrain/imperative/ops/autogen.h"
#include <vector>
#include <unordered_map>
#include <string>
#include "megbrain/serialization/serializer.h"
#include "./utils.h"
#include "./engine.h"

using namespace mgb::imperative::interpreter;
namespace mgb::imperative::js {

void readfs(const char* fname){
    FILE* fp = fopen(fname, "rt");
    if(fp){
        while(!feof(fp)){
            char c = fgetc(fp);
            if(c != EOF){
                putchar(c);
            }
        }
        putchar('\n');
        fclose(fp);
    }
    else{
        std::cout << "unable to open file: "<<fname<<std::endl;
    }
}


class ModelExecutor{
public:
    ModelExecutor(std::string model_path);
    
#ifdef __EMSCRIPTEN__
    int32_t forward(const emscripten::val &v, const emscripten::val &data, const int type);

#else 
    int32_t forward(){
        throw std::runtime_error("forward not implemented");
    }
#endif

    std::string getInputShape(){
        return network.tensor_map.begin()->second->shape().to_string();
    }

private:
    serialization::GraphLoader::LoadResult network;
    std::unordered_map<int, std::shared_ptr<Tensor>> tensor_registry;
    int output_id;
};


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_BINDINGS(model_executor) {
    emscripten::class_<ModelExecutor>("WasmModelExecutor")
        .constructor<std::string>()
        .function("forward", &ModelExecutor::forward)
        .function("getInputShape", &ModelExecutor::getInputShape)
        ;
}
#endif

}
