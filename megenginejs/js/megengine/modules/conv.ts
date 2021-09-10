import {ENGINE} from "../engine";
import {Tensor} from "../tensor";
import {Module} from "./module";

class _Conv2D extends Module{
    in_channels: number
    out_channels: number
    kernel_size: number
    stride: number
    padding: number
    weight: Tensor
    bias: Tensor

    constructor (
        in_channels: number, 
        out_channels: number,
        kernel_size: number,
        stride: number,
        padding: number,
        bias: boolean = true
        ){
        super();
        this.in_channels = in_channels
        this.out_channels = out_channels
        this.kernel_size = kernel_size
        this.stride = stride
        this.padding = padding
        
        const fanin = this._get_fanin();

        this.weight = ENGINE.rand(this._infer_weight_shape(), 0.0, Math.sqrt(1.0 / fanin));
        // console.log(ENGINE.getTensorShape(this.weight.data));
        if(bias){
            this.bias = ENGINE.zeros(this._infer_bias_shape());
        }
    }
    
    forward(inp: Tensor): Tensor{
        let outID = ENGINE.wasm.conv2d(
            inp.data, this.weight.data, 
            this.stride, this.padding);
        let offset = ENGINE.getMemOffset(outID, inp.dtype);
        let out = new Tensor(outID, ENGINE.getTensorShape(outID), offset);
        ENGINE.track(out);
        ENGINE.tensorMap.set(outID, out)
        if(this.bias){
            out.add_(this.bias);
        }
        return out;
    }

    _get_fanin(): number{
        return this.kernel_size * this.kernel_size * this.in_channels
    }

    _infer_weight_shape(){
        const ichl = this.in_channels
        const ochl = this.out_channels
        const kernel = this.kernel_size
        return [ochl, ichl, kernel, kernel]
    }

    _infer_bias_shape(){
        return [1, this.out_channels, 1, 1]
    }
    
}

export function Conv2D(        
    in_channels: number, 
    out_channels: number,
    kernel_size: number,
    stride: number = 1,
    padding: number = 0,
    bias: boolean = true): _Conv2D{

    return new _Conv2D(in_channels, out_channels, kernel_size, stride, padding, bias);

}