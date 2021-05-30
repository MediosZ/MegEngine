#include "megbrain/webassembly.h"
#include "megbrain/serialization/serializer.h"
#include <iostream>
using namespace mgb;

cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                      HostTensorND& host) {
    auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
    return {dev, cb};
}

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


extern "C"{


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
int generate(){
    std::cout << "generate some number: " << 42 << std::endl;
    return 142;
}


#ifdef __EMSCRIPTEN__
EMSCRIPTEN_KEEPALIVE
#endif
void runProgram(){
    std::cout << "runProgram" << std::endl;
    readfs("test.txt");
    
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs("xornet_deploy.mge");
    std::cout << "make fs" << std::endl;
    auto loader = serialization::GraphLoader::make(std::move(inp_file));
    serialization::GraphLoadConfig config;
    serialization::GraphLoader::LoadResult network =
            loader->load(config, false);
    std::cout << "load network" << std::endl;
    auto data = network.tensor_map["data"];
    float* data_ptr = data->resize({1, 2}).ptr<float>();
    data_ptr[0] = 0.6;
    data_ptr[1] = 0.9;
    HostTensorND predict;
    std::unique_ptr<cg::AsyncExecutable> func =
            network.graph->compile({make_callback_copy(
                    network.output_var_map.begin()->second, predict)});
    std::cout << "before execute" << std::endl;
    func->execute();
    func->wait();
    float* predict_ptr = predict.ptr<float>();
    std::cout << " Predicted: " << predict_ptr[0] << " " << predict_ptr[1]
              << std::endl;
}

}
