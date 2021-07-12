import { MegEngine, WasmFactoryConfig  } from "../../wasm-out/meg";
import Generator from "../../wasm-out/meg.js";

export { MegEngine } from "../../wasm-out/meg";

export function init(): Promise<MegEngine> {
  return new Promise((resolve, reject) => {
      const config: WasmFactoryConfig = {};
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
