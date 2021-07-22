import {WasmEngine} from "../js/megengine/wasm_engine";

export interface MegEngine extends EmscriptenModule {
    Engine: typeof WasmEngine
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
