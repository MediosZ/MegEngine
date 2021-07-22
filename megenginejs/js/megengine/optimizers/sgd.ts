import {ENGINE} from "../engine";
import {Tensor} from "../tensor";
import {BaseOptimizer} from "./base";

class _SGD extends BaseOptimizer{
    constructor(parameters: Tensor[], learning_rate: number){
        super(parameters, learning_rate);
    }

    update(t: Tensor, grad: Tensor){
        ENGINE.tidy(() => {
            t.sub_(grad.mul(this.learning_rate));
        });
    }
}

export function SGD(parameters: Tensor[], learning_rate: number){
    return new _SGD(parameters, learning_rate);
}