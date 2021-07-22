import {Tensor} from "../tensor";

export class BaseOptimizer{
    learning_rate: number
    parameters: Tensor[]

    constructor(parameters: Tensor[], learning_rate: number){
        this.learning_rate = learning_rate;
        this.parameters = parameters;
    }

    step(){
        for(let tensor of this.parameters){
            this.update(tensor, tensor.grad);
        }
    }

    update(t: Tensor, grad: Tensor){
        throw Error("Not implemented");
    }
}