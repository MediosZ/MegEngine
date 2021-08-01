
import {ENGINE, setWasmPath, createModelExecutor, DType, GradManager, Tensor} from "megenginejs";
import wasmPath from "megenginejs/meg.wasm";
import model from "./xornet_deploy.mge";

async function run() {
    setWasmPath(wasmPath);
    console.log("Model Executor!");
    await ENGINE.init();
    console.log("Model Executor initialized!");
    /*
    let a = ENGINE.rand([20, 20]);
    let b = ENGINE.rand([20, 20]);
    ENGINE.printTensor(a);
    let t1 = ENGINE.rand([50, 1, 28, 28]);
    let t2 = ENGINE.rand([6, 1, 5, 5]);
    let bias = ENGINE.rand([1, 6, 1, 1]);
    let gm = new GradManager();
    gm.attach([a,b]);
    gm.backward(() => {
        let outID = ENGINE.engine.conv2d(
            t1.data, t2.data, 
            1, 0);
        let offset = ENGINE.getMemOffset(outID, t1.dtype);
        let out = new Tensor(outID, ENGINE.getTensorShape(outID), offset);
        ENGINE.track(out);
        ENGINE.tensorMap.set(outID, out)
        out.add_(bias);
        
        let out = ENGINE.matmul(a, b);
        // out = ENGINE.reshape(out, [40, 10]);
        // console.log(ENGINE.getTensorShape(out.data));
        return out;
    });
    console.log("finish");
    */
    let executor = await createModelExecutor(model);
    let out = executor.forward([0.6, 0.9, 0.6, 0.9], {shape: [2, 2], dtype: DType.float32});
    
    ENGINE.printTensor(out);
    ENGINE.cleanup();
}

run();
