import {DType, RecursiveArray, TypedArray} from './dtypes';
import {inferShape, inferSizeFromShape, flatten} from './utils';
import {init, MegEngine, setWasmPath} from './backend';
import {Tensor} from './tensor';
import { isThisTypeNode } from 'typescript';

export {setWasmPath} from './backend';

class Engine{
  tensorMap: Map<number, Tensor>
  wasm: MegEngine
  curId: number
  engine: object
  opRegistry: Map<string, Function>

  constructor(){
      this.curId = 0;
      // this.wasm = wasm;
      this.tensorMap = new Map<number, Tensor>();
  }

  async init(){
    this.wasm = await init();
    this.wasm.ccall("initTensor", null, null, null);
    this.registerTensor = this.wasm.cwrap('registerTensor', 'number',
      ['number', 'number', 'array']);
    this.engine = new this.wasm.Engine();
  }

  apply(){
    this.wasm.jsapply();
  }
  testBackward(){

    this.wasm.ccall("jsbackward", null, null, null);
    // this.wasm.testBackward();
  }

  tensor(data: RecursiveArray<number>, shape?:number[]): Tensor{
      let inferedShape = shape || inferShape(data);
      const shapeBytes = new Int32Array(inferedShape);
      data = flatten(data);
      let arrayBuffer = new Float32Array(data);

      let id = this.engine.registerTensor(shapeBytes, arrayBuffer);
      let memOffset = this.getMemOffset(id);
      let tensor = new Tensor(id, inferedShape, memOffset, DType.float32);
      this.tensorMap.set(id, tensor);
      return tensor;
  }

  rand(shape: number[], mean: number = 0.0, std: number = 1.0): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.randn(shapeBytes, mean, std);
    let offset = this.getMemOffset(outid);
    let out = new Tensor(outid, shape, offset, DType.float32);
    this.tensorMap.set(outid, out);
    return out;
  }

  zeros(shape: number[]): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.zeros(shapeBytes);
    let offset = this.getMemOffset(outid);
    let out = new Tensor(outid, shape, offset, DType.float32);
    this.tensorMap.set(outid, out);
    return out;
  }

  ones(shape: number[]): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.ones(shapeBytes);
    let offset = this.getMemOffset(outid);
    let out = new Tensor(outid, shape, offset, DType.float32);
    this.tensorMap.set(outid, out);
    return out;
  }

  getMemOffset(id: number): number{
    return this.engine.getTensorOffset(id);
  }

  getTensorShape(id: number): number[]{
    let shapeString = this.engine.getTensorShape(id);
    let shape = shapeString.substring(1, shapeString.length-1).split(",").map((x: string) => parseInt(x));
    // console.log(shape);
    return shape;
}

  async read(t: Tensor): Promise<TypedArray>{
    return this.readSync(t);
  }

  readSync(tensor: Tensor): TypedArray{
    const offset = this.getMemOffset(tensor.data);
    // console.log(offset);
    const bytes = this.wasm.HEAPU8.slice(
      offset,
      offset + inferSizeFromShape(tensor.shape) * 4);
    return new Float32Array(bytes.buffer);
  }

  updateGrad(t: Tensor){
    const grad_id = this.engine.getGrad(t.data);
    const grad_offset = this.engine.getGradOffset(t.data);
    t.grad = new Tensor(grad_id, t.shape, grad_offset, t.dtype);
  }

  printTensor(t: Tensor, msg?: string){
    let info = msg || "";
    console.log(info + this.readSync(t));
  }

  printGrad(t: Tensor, msg?: string){
    if(!t.grad){
      throw `Tensor ${t.data} has no grad`;
    }
    let info = msg || "";
    console.log(info + this.readSync(t.grad));
  }

  disposeTensor(t: Tensor){
    this.engine.disposeTensor(t.data);
  }

  applyOp(opName: string, ...tensors: Tensor[]){
      let op = this.opRegistry.get(opName);
    
  }

  add(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.add(tensorA.data, tensorB.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sub(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.sub(tensorA.data, tensorB.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sub_(a: Tensor, b: Tensor){
    this.engine.sub_(a.data, b.data);
  }

  add_(a: Tensor, b: Tensor){
    this.engine.add_(a.data, b.data);
  }

  mul(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.mul(tensorA.data, tensorB.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  div(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.div(tensorA.data, tensorB.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }


  matmul(a: Tensor, b: Tensor, transposeA: boolean = false, transposeB: boolean = false): Tensor{
    let outID = this.engine.matmul(a.data, b.data, transposeA, transposeB);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sin(a: Tensor): Tensor{
    let outID = this.engine.sin(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }


  log(a: Tensor): Tensor{
    let outID = this.engine.log(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  cos(a: Tensor): Tensor{
    let outID = this.engine.cos(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  mean(a: Tensor, axis?: number, keepdims=false): Tensor{
    return this.reduce(5)(a, axis, keepdims);
  }
  max(a: Tensor, axis?: number, keepdims=false): Tensor{
    return this.reduce(4)(a, axis, keepdims);
  }
  min(a: Tensor, axis?: number, keepdims=false): Tensor{
    return this.reduce(3)(a, axis, keepdims);
  }
  sum(a: Tensor, axis?: number, keepdims=false): Tensor{
    return this.reduce(0)(a, axis, keepdims);
  }

  square(a: Tensor): Tensor{
    return this.mul(a, a);
  }

  relu(a: Tensor): Tensor{
    let outID = this.engine.relu(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  exp(a: Tensor): Tensor{
    let outID = this.engine.exp(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  reshape(a: Tensor, shape: number[]){
      let unspec_axis: number = undefined;
      shape.forEach((value, index) => {
        if(value < 0){
            if(value !== -1){
                throw Error(`expect shape[${index}] >= -1, got ${value}`);
            }
            if(unspec_axis){
                throw Error(`multiple -1 in shape: ${unspec_axis} and ${idx}`);
            }
            unspec_axis = index;
        }
      });

      let outID = this.engine.reshape(a.data, shape, (unspec_axis === undefined) ? -1 : unspec_axis);
      let offset = this.getMemOffset(outID);
      let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
      this.tensorMap.set(outID, out);
      return out;
  }

  removeAxis(a: Tensor, axis: number[]){
    let ax = axis.map((value, index) => {
        return value - index;
    });
    let outID = this.engine.removeAxis(a.data, ax);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out; 
  }

  addAxis(a: Tensor, axis: number[]){
    let outID = this.engine.addAxis(a.data, axis);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out; 
  }

  squeeze(t: Tensor, axis?: number){
    if(axis === undefined || axis === -1){
      return this.removeAxis(t, [t.shape.length - 1]);
    }
    else{
      return this.removeAxis(t, [axis]);
    }
  }

  unsqueeze(t: Tensor, axis?: number){
    if(axis === undefined || axis === -1){
      return this.addAxis(t, [t.shape.length]);
    }
    else{
      return this.addAxis(t, [axis]);
    }
  }


  reduce(mode: number){
    return (a: Tensor, axis: null | number, keepdims: boolean = false) => {
        if(axis === undefined){
            let fla = this.reshape(a, [-1]);
            if(keepdims){
                throw Error("can not set axis=null and keepdims=true");
            }
            let outID = this.engine.reduce(fla.data, mode, 0);
            let offset = this.getMemOffset(outID);
            let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
            this.tensorMap.set(outID, out);
            return out;
        }
        else{
            let outID = this.engine.reduce(a.data, mode, axis);
            let offset = this.getMemOffset(outID);
            let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
            this.tensorMap.set(outID, out);
            if(!keepdims){
                let axis: number[] = [];
                out.shape.forEach((element, index) => {
                    if(element == 1){
                        axis.push(index);
                    }
                });
                return this.removeAxis(out, axis);
            }
            return out;
        }
    }
  }

  flattern(t: Tensor){
      return this.reshape(t, [-1]);
  }

  index_one_hot(t: Tensor, index: number[], axis: number = 1, keepdims: boolean = false){
    let outID = this.engine.index_one_hot(t.data, index, axis);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, this.getTensorShape(outID), offset, DType.float32);
    this.tensorMap.set(outID, out);
    if(keepdims){
        return out;
    }
    else{
        return this.squeeze(out, axis);
    }

  }

  logsumexp(t: Tensor, axis: number, keepdims: boolean = false){
      let maxValue = this.max(t, axis, true);
    if(keepdims){
        return maxValue.add(this.log(
            this.sum(this.exp(t.sub(maxValue)), axis, keepdims)
        ));
    }
    else{
        return this.squeeze(maxValue).add(this.log(
            this.sum(this.exp(t.sub(maxValue)), axis, keepdims)
        ));
    }
  }

  cleanup(){
      this.engine.delete();
  }
}


let globalNameSpace : {megGlobals: Map<string, any>};

export function getGlobalNameSpace() : {megGlobals: Map<string, any>}{
  if (globalNameSpace == null) {
    // tslint:disable-next-line:no-any
    let ns: any = window;
    globalNameSpace = ns;
  }
  return globalNameSpace;
}

export function getOrInitEngine(): Engine{
  const namespace = getGlobalNameSpace() as {} as {megengine: Engine};
  if (namespace.megengine == null) {
    namespace.megengine = new Engine();
  }
  return namespace.megengine;
}

export const ENGINE = getOrInitEngine();
