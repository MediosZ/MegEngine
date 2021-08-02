

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
    
    replaceTensor(arg0: Int, arg1: any, arg2: any, arg3: Int): Int;

    registerTensor(arg0: any, arg1: any, arg2: Int): Int;
    
    zeros(arg0: any, arg1: Int): Int;
    
    ones(arg0: any, arg1: Int): Int;
    
    disposeTensor(arg0: Int): void;
    
    getTensorOffset(arg0: Int, arg1: Int): Int;
    
    getGradOffset(arg0: Int, arg1: Int): Int;
    
    getGrad(arg0: Int): Int;
    
    randn(arg0: any, arg1: Float, arg2: Float): Int;
    
    mul(arg0: Int, arg1: Int): Int;
    
    div(arg0: Int, arg1: Int): Int;
    
    matmul(arg0: Int, arg1: Int, arg2: boolean, arg3: boolean): Int;
    
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
    
    log(arg0: Int): Int;
    
    reduce(arg0: Int, arg1: Int, arg2: Int): Int;
    
    removeAxis(arg0: Int, arg1: any): Int;
    
    addAxis(arg0: Int, arg1: any): Int;
    
    index_one_hot(arg0: Int, arg1: Int, arg2: Int): Int;
    
    exp(arg0: Int): Int;
    
    getTensorShape(arg0: Int): string;
    
    astype(arg0: Int, arg1: Int): Int;
    
    argmax(arg0: Int, arg1: Int): Int;
    delete(): void;
}

