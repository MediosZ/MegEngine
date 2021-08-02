
import wasmPath from "megenginejs/meg.wasm";

import {ENGINE, setWasmPath, GradManager, SGD, MSE, Linear} from "megenginejs";

async function run() {
    setWasmPath(wasmPath);
    console.log("Playground!");
    await ENGINE.init();

    let tensor = ENGINE.rand([3, 4, 3]);
    tensor.print();
    ENGINE.cleanup();
}

run();
