
import wasmPath from "megenginejs/meg.wasm";

import * as mge from "megenginejs";

async function run() {
  mge.setWasmPath(wasmPath);
  console.log("Playground!");
  mge.run(async () => {
    let tensor = mge.rand([3, 4, 3]);
    tensor.print();
  });
}

run();
