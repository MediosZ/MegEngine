import wasmPath from "megenginejs/meg.wasm";
import * as mge from "megenginejs";
import model from "./xornet_deploy.mge";

async function run() {
  console.log("Model Executor!");
  mge.setWasmPath(wasmPath);
  mge.run(async () => {
    let executor = await mge.createModelExecutor(model);
    let out = executor.forward([0.6, 0.9, 0.6, 0.9], {shape: [2, 2], dtype: mge.DType.float32});
    out.print();
  });
}

run();
