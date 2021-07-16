import {Tensor} from './tensor';
import {ENGINE} from './engine';


export class GradManager{
    attached_tensors: Tensor[]
    constructor(){
      this.attached_tensors = []
    }
  
    attach(tensors: Tensor[]): GradManager {
      for(let tensor of tensors){
        this.attached_tensors.push(tensor);
      }
      return this;
    }
  
    backward(compute: ()=>Tensor){
      let engine = ENGINE.engine;
      engine.startScope();
  
      // attach all tensors
      for(let tensor of this.attached_tensors){
        engine.attach(tensor.data);
      }
      let loss = compute();
  
      // backward
      engine.backward(loss.data);

      for(let tensor of this.attached_tensors){
        ENGINE.updateGrad(tensor);
      }
      //clean up
      engine.endScope();
    }


    /*
    optimize(lr: Tensor, bs: Tensor){
      for(let tensor of this.attached_tensors){
        ENGINE.updateGrad(tensor);
        tensor.sub_(tensor.grad.mul(lr).div(bs));
        // ENGINE.sub_(tensor, ENGINE.div(ENGINE.mul(tensor.grad, lr), bs));
        // ENGINE.printTensor(tensor, `t<${tensor.data}>: `);
      }
    }
    */
  }