
import wasmPath from "megenginejs/meg.wasm";

import * as mge from "megenginejs";

async function run() {
  try {
    mge.setWasmPath(wasmPath);
    console.log("Playground!");
    await mge.init();

    let tensor = mge.rand([3, 4, 3]);
    tensor.print();
  }
  finally{ 
    mge.cleanup();
  }
}

run();
