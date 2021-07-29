#include "./model_executor.h"

namespace mgb::imperative::js {


cg::ComputingGraph::OutputSpecItem make_callback_copy(SymbolVar dev,
                                                      HostTensorND& host) {
    auto cb = [&host](DeviceTensorND& d) { host.copy_from(d); };
    return {dev, cb};
}
ModelExecutor::ModelExecutor(std::string path){
    std::unique_ptr<serialization::InputFile> inp_file =
            serialization::InputFile::make_fs(path.c_str());
    auto loader = serialization::GraphLoader::make(std::move(inp_file));
    serialization::GraphLoadConfig config;
    network = loader->load(config, false);
    tensor_registry = std::unordered_map<int, std::shared_ptr<Tensor>>();
    // auto engine = EngineWrapperInst();
    output_id = -1;
}

#ifdef __EMSCRIPTEN__
int32_t ModelExecutor::forward(const emscripten::val &v, const emscripten::val &data, const int type = 0){
    auto rv = getVectorFromVal(v);
    TensorShape shape = TensorShape(rv);
    
    auto cn = CompNode::load("cpu0");
    std::shared_ptr<HostTensorND> ret = std::make_shared<HostTensorND>(cn, shape, getDataType(type));
    assignData(data, ret, type);

    auto& net_data = network.tensor_map.begin()->second;
    net_data->resize(shape).copy_from(*ret);

    HostTensorND predict;
    std::unique_ptr<cg::AsyncExecutable> func =
            network.graph->compile({make_callback_copy(
                    network.output_var_map.begin()->second, predict)});
    func->execute();
    func->wait();
    /*
    auto t_out_value = predict.ptr<float>();
    for(size_t i = 0; i < predict.shape().total_nr_elems(); i++){
        mgb_log("value<%d>: %f", i, t_out_value[i]);
    }
    */
    auto handle = interpreter_for_js->put(predict, true);
    auto outTensor = std::make_shared<Tensor>(handle);
    if(output_id == -1){        
        output_id = EngineWrapper::Inst()->registerTensor(outTensor);
    }
    else{
        EngineWrapper::Inst()->replaceTensor(output_id, outTensor);
    }

    return output_id;
}

#endif

}