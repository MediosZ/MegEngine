import { MegEngine, WasmFactoryConfig  } from "../../wasm-out/meg";
import Generator from "../../wasm-out/meg.js";
// import {wasmWorkerContents} from "../../wasm-out/meg.worker.js";
export { MegEngine } from "../../wasm-out/meg";
let wasmPath: string = null;
export let customFetch: Function = undefined;

export function setWasmPath(path: string, custom?: Function) {
    wasmPath = path;
    customFetch = custom;
}

function createInstantiateWasmFunc(path: string) {
    // tslint:disable-next-line:no-any
    return (imports: any, callback: any) => {
      customFetch(path).then((response: any) => {
        if (!response['ok']) {
          imports.env.a(`failed to load wasm binary file at '${path}'`);
        }
        response.arrayBuffer().then((binary: any) => {
          WebAssembly.instantiate(binary, imports).then(output => {
            callback(output.instance, output.module);
          });
        });
      });
      return {};
    };
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
      if (customFetch!==undefined) {
        config.instantiateWasm = createInstantiateWasmFunc(wasmPath);
      }
      let wasm: Promise<MegEngine> = Generator(config);
      wasm.then((module) => {
          resolve(module);
      })
  })
}
