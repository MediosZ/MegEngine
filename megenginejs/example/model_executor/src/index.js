
import {ENGINE, setWasmPath, createModelExecutor, DType, GradManager, Tensor} from "megenginejs";
import wasmPath from "megenginejs/meg.wasm";
import model from "./xornet_deploy.mge";

async function run() {
    setWasmPath(wasmPath);
    console.log("Model Executor!");
    await ENGINE.init();
    
    let executor = await createModelExecutor(model);
    let out = executor.forward([0.6, 0.9, 0.6, 0.9], {shape: [2, 2], dtype: DType.float32});
    
    ENGINE.printTensor(out);
    ENGINE.cleanup();
}

run();
