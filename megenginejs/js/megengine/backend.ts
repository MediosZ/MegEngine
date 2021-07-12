import { MegEngine, WasmFactoryConfig  } from "../../wasm-out/meg";
import Generator from "../../wasm-out/meg.js";

export { MegEngine } from "../../wasm-out/meg";
let wasmPath: string = null;

export function setWasmPath(path: string){
    wasmPath = path;
}

export function init(): Promise<MegEngine> {
  return new Promise((resolve, reject) => {
      const config: WasmFactoryConfig = {
          locateFile: (path, prefix) => {
            if (path.endsWith('.wasm')) {
                if(wasmPath !== null){
                    return wasmPath;
                }
                let path = "meg.wasm";
                return prefix + path;
            }
            return prefix + path;
          }
      };
      let wasm: Promise<MegEngine> = Generator(config);
      wasm.then((module) => {
          /*
          module.generator = {
              generate: module.cwrap("generate", 'number', null),
              runProgram: module.cwrap("runProgram", null, null)
          }
          */
          resolve(module);
      })
  })
}
