
import wasmPath from "megenginejs/meg.wasm";
import model from "./xornet_deploy.mge";

import {ENGINE, setWasmPath, createModelExecutor, DType} from "megenginejs";

async function run() {
    setWasmPath(wasmPath);
    console.log("Model Executor!");
    await ENGINE.init();
    let me = await createModelExecutor(model);
    me.forward([0.6, 0.9, 0.6, 0.9], {shape: [2, 2], dtype: DType.float32});
    
    ENGINE.cleanup();
}

run();
