import {ENGINE} from "../engine";
import {Tensor} from "../tensor";
import {Module} from "./module";

class _Pool2D extends Module{
    kernel_size: number
    stride: number
    padding: number
    mode: number
    
    constructor(kernel_size: number, stride: number, padding: number, mode: number){
        super();
        this.kernel_size = kernel_size;
        this.stride = stride;
        this.padding = padding;
        this.mode = mode;
    }

    forward(inp: Tensor){
        let outID = ENGINE.engine.pool(
            inp.data, this.kernel_size, this.stride, this.padding, this.mode);
        let offset = ENGINE.getMemOffset(outID);
        let out = new Tensor(outID, ENGINE.getTensorShape(outID), offset);
        ENGINE.track(out);
        ENGINE.tensorMap.set(outID, out);
        return out;
    }
}

class _MaxPool2D extends _Pool2D{
    constructor(kernel_size: number, stride: number, padding: number){
        super(kernel_size, stride, padding, 0);
    }
    
}

class _AveragePool2D extends _Pool2D{
    constructor(kernel_size: number, stride: number, padding: number){
        super(kernel_size, stride, padding, 1);
    }
}

export function MaxPool2D(kernel_size: number, stride?: number, padding?: number) {
    return new _MaxPool2D(kernel_size, stride || kernel_size, padding || 0);
}

export function AveragePool2D(kernel_size: number, stride?: number, padding?: number) {
    return new _AveragePool2D(kernel_size, stride || kernel_size, padding || 0);
}
