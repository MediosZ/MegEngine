import {ENGINE} from "../engine";
import {Tensor} from "../tensor";

export async function init(){
    await ENGINE.init();
} 

export function cleanup(){
  ENGINE.cleanup();
}

export function tidy(callback: Function): Tensor{
  return ENGINE.tidy(callback);
}

export function disposeTensor(tensor: Tensor): void{
  ENGINE.disposeTensor(tensor);
}

export function printTensor(tensor: Tensor, msg?: string){
  ENGINE.printTensor(tensor, msg);
}