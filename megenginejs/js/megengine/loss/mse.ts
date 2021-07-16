import { Tensor } from "../tensor";
import { ENGINE } from "../engine";

export function MSE(inp: Tensor, target: Tensor): Tensor{
    return inp.sub(target).square().mean();
}