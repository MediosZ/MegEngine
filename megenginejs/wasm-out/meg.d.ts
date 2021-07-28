import { TypedArray } from "../js/megengine";
import { WasmEngine, WasmModelExecutor } from "../js/megengine/wasm_dtypes";

export interface MegEngine extends EmscriptenModule {
    Engine: typeof WasmEngine;
    WasmModelExecutor: typeof WasmModelExecutor;
    FS_createDataFile(parent: string, path: string, data: TypedArray, canread: boolean, canwrite: boolean, canown: boolean): any;
}

export interface WasmFactoryConfig {
  mainScriptUrlOrBlob?: string|Blob;
  locateFile?(path: string, prefix: string): string;
  instantiateWasm?: Function;
  onRuntimeInitialized?: () => void;
  onAbort?: (msg: string) => void;
}

declare var moduleFactory: (settings: WasmFactoryConfig) =>
    Promise<MegEngine>;
export default moduleFactory;
