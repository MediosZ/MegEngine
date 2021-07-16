import { Tensor } from "../tensor";
import { ENGINE } from "../engine";

export function CrossEntropy(
    inp: Tensor, 
    target: number[],
    axis: number = 1,
    with_logits: boolean = true
    ): Tensor{
    if(with_logits){
        return ENGINE.logsumexp(inp, axis).mean().sub(
            ENGINE.index_one_hot(inp, target, axis).mean()
        );
    }
    else{
        return ENGINE.zeros([1]).sub(ENGINE.log(ENGINE.index_one_hot(inp, target, axis)).mean());
    }
}