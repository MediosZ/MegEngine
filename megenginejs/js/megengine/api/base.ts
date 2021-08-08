import {ENGINE} from "../engine";
import {Tensor} from "../tensor";


export async function run(fn: Function){
  try{
    await init();
    await fn();
  }
  catch(e){
    throw new Error(e);
  }
  finally{
    cleanup();
  }
}

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


const delayCallback: Function = (() => {
  if (typeof requestAnimationFrame !== 'undefined') {
    return requestAnimationFrame;
  } else if (typeof setImmediate !== 'undefined') {
    return setImmediate;
  }
  return (f: Function) => f();  // no delays
})();

export function nextFrame(): Promise<void> {
  return new Promise<void>(resolve => delayCallback(() => resolve()));
}
