import { TypedArray } from "../js/megengine";
import { WasmEngine, WasmModelExecutor } from "../js/megengine/wasm_dtypes";
type Int = number;
type Float = number;

export interface MegEngine extends EmscriptenModule {
    Engine: typeof WasmEngine;
    WasmModelExecutor: typeof WasmModelExecutor;
    FS_createDataFile(parent: string, path: string, data: TypedArray, canread: boolean, canwrite: boolean, canown: boolean): any;
    initTensor(): void;
    inst(): WasmEngine;

    randn(arg0: any, arg1: Float, arg2: Float): Int;
    zeros(arg0: any, arg1: Int): Int;
    ones(arg0: any, arg1: Int): Int;
    eq(arg0: Int, arg1: Int): Int;
    dot(arg0: Int, arg1: Int): Int;
    mul(arg0: Int, arg1: Int): Int;
    div(arg0: Int, arg1: Int): Int;
    matmul(arg0: Int, arg1: Int, arg2: boolean, arg3: boolean): Int;
    batch_matmul(arg0: Int, arg1: Int, arg2: boolean, arg3: boolean): Int;
    add(arg0: Int, arg1: Int): Int;
    sub(arg0: Int, arg1: Int): Int;
    add_(arg0: Int, arg1: Int): Int;
    sub_(arg0: Int, arg1: Int): Int;
    sin(arg0: Int): Int;
    cos(arg0: Int): Int;
    conv2d(arg0: Int, arg1: Int, arg2: Int, arg3: Int): Int;
    pool(arg0: Int, arg1: Int, arg2: Int, arg3: Int, arg4: Int): Int;
    relu(arg0: Int): Int;
    reshape(arg0: Int, arg1: any, arg2: Int): Int;
    broadcast_to(arg0: Int, arg1: any): Int;
    log(arg0: Int): Int;
    reduce(arg0: Int, arg1: Int, arg2: Int): Int;
    removeAxis(arg0: Int, arg1: any): Int;
    addAxis(arg0: Int, arg1: any): Int;
    index_one_hot(arg0: Int, arg1: Int, arg2: Int): Int;
    exp(arg0: Int): Int;
    astype(arg0: Int, arg1: Int): Int;
    argmax(arg0: Int, arg1: Int): Int;
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
