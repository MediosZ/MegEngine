import wasmPath from "megenginejs/meg.wasm";
import * as mge from "megenginejs";
import model from "./xornet_deploy.mge";

async function run() {
  try{
    mge.setWasmPath(wasmPath);
    console.log("Model Executor!");
    await mge.init();
    
    let executor = await mge.createModelExecutor(model);
    let out = executor.forward([0.6, 0.9, 0.6, 0.9], {shape: [2, 2], dtype: mge.DType.float32});
    
    out.print();
  }
  finally{
    mge.cleanup();
  }
}

run();
