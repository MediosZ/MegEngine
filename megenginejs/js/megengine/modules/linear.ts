import {ENGINE} from "../engine";
import {Tensor} from "../tensor";
import {Module} from "./module";

class _Linear extends Module{
    in_channels: number
    out_channels: number
    bias: Tensor
    weight: Tensor

    constructor(in_channels: number, out_channels: number, bias: boolean = true){
        super();
        this.in_channels = in_channels;
        this.out_channels = out_channels;

        this.weight = ENGINE.rand([this.out_channels, this.in_channels], 0, Math.sqrt(1.0 / this.in_channels));
        if(bias){
            this.bias = ENGINE.zeros([this.out_channels]);
        }
    }

    forward(inp: Tensor): Tensor{
        let out = ENGINE.matmul(inp, this.weight, false, true);
        if(this.bias){
            out.add_(this.bias);
        }
        return out;
    }
}

export function Linear(in_channels: number, out_channels: number, bias: boolean = true) {
    return new _Linear(in_channels, out_channels, bias);
}