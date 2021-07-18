import { Tensor } from "../tensor";
import { ENGINE } from "../engine";

export function MSE(inp: Tensor, target: Tensor): Tensor{
    return ENGINE.tidy(() =>{
        return inp.sub(target).square().mean();
    });
}