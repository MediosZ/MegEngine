

type Int = number;
type Float = number;

export class WasmModelExecutor {
    constructor(arg0: string);
    forward(arg0: any, arg1: any, arg2: Int): Int;
    getInputShape(): string;
    delete(): void;
}

export class WasmEngine {
    constructor();
    // static inst(): WasmEngine;
    size(): Int;
    startScope(): void;
    endScope(): void;
    attach(arg0: Int): void;
    backward(arg0: Int): void;
    replaceTensorWithData(arg0: Int, arg1: any, arg2: any, arg3: Int): Int;
    replaceTensorWithID(arg0: Int, arg1: Int): Int;
    registerTensor(arg0: any, arg1: any, arg2: Int): Int;
    disposeTensor(arg0: Int): void;
    getTensorOffset(arg0: Int, arg1: Int): Int;
    getGradOffset(arg0: Int, arg1: Int): Int;
    getGrad(arg0: Int): Int;
    getTensorShape(arg0: Int): string;
    delete(): void;
}

