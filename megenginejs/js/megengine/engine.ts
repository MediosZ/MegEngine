import {DType, RecursiveArray, TypedArray, isTypedArray} from './dtypes';
import {inferShape, inferSizeFromShape, flatten} from './utils';
import {init, MegEngine } from './backend';
import {Tensor, TensorInfo} from './tensor';
export {setWasmPath} from './backend';
import {WasmEngine} from "./wasm_dtypes";

interface ScopeState {
    track: Tensor[];
    name: string;
    id: number;
}

class Engine{
  tensorMap: Map<number, Tensor>
  wasm: MegEngine
  engine: WasmEngine
  opRegistry: Map<string, Function>
  scopeStack: ScopeState[]
  activeScope: ScopeState
  nextScopeId: number

  constructor(){
      this.tensorMap = new Map<number, Tensor>();
      this.scopeStack = [];
      this.nextScopeId = 0;
  }

  async init(){
    this.wasm = await init();
    this.wasm.initTensor();
    this.engine = this.wasm.inst(); // = new this.wasm.Engine();
  }

  size(): number{
      return this.engine.size();
  }

  tensor(data: RecursiveArray<number> | TypedArray, info :TensorInfo = {dtype: DType.float32}): Tensor{
      let {shape, dtype} = info;
      let id;
      let inferedShape;
      if(isTypedArray(data)){
        inferedShape = shape || [(data as TypedArray).length];
        const shapeBytes = new Int32Array(inferedShape);
        id = this.engine.registerTensor(shapeBytes, data, dtype);
      }
      else{
        inferedShape = shape || inferShape(data);
        const shapeBytes = new Int32Array(inferedShape);
        let arrayBuffer = new Float32Array(flatten(data));
        id = this.engine.registerTensor(shapeBytes, arrayBuffer, dtype);
      }
      return this.createTensor(id, inferedShape, dtype);
  }

  createTensor(id: number, shape: number[], dtype: DType): Tensor{
    let memOffset = this.getMemOffset(id, dtype);
    let tensor = new Tensor(id, shape, memOffset, dtype);
    this.track(tensor);
    this.tensorMap.set(id, tensor);
    return tensor;
  }

  track(t: Tensor): Tensor{
    if (this.activeScope != null) {
        t.scopeid = this.activeScope.id;
        this.activeScope.track.push(t);
      }
  
      return t;
  }

  rand(shape: number[], mean: number = 0.0, std: number = 1.0, dtype: DType = DType.float32): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.randn(shapeBytes, mean, std);
    return this.createTensor(outid, shape, dtype);
  }

  zeros(shape: number[], dtype: DType = DType.float32): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.zeros(shapeBytes, dtype);
    return this.createTensor(outid, shape, dtype);
  }

  ones(shape: number[], dtype: DType = DType.float32): Tensor{
    let inferedShape = shape;
    const shapeBytes = new Int32Array(inferedShape);
    let outid = this.engine.ones(shapeBytes, dtype);
    return this.createTensor(outid, shape, dtype);
  }

  getMemOffset(id: number, dtype: DType = DType.float32): number{
    return this.engine.getTensorOffset(id, dtype);
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
    const offset = this.getMemOffset(tensor.data, tensor.dtype);
    // console.log(offset);
    const bytes = this.wasm.HEAPU8.slice(
      offset,
      offset + inferSizeFromShape(tensor.shape) * 4);
    if(tensor.dtype == DType.float32){
        return new Float32Array(bytes.buffer);
    }
    else if(tensor.dtype == DType.int32){
        return new Int32Array(bytes.buffer);
    }
    else if(tensor.dtype == DType.int8){
        return new Int8Array(bytes.buffer);
    }
    else {
        return new Uint8Array(bytes.buffer);
    }
    
  }

  updateGrad(t: Tensor){
    const grad_id = this.engine.getGrad(t.data);
    if(grad_id === -1){
        // tensor has no grad, return 
        return;
    }
    const grad_offset = this.engine.getGradOffset(t.data, t.dtype);
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
    this.tensorMap.delete(t.data);
    this.engine.disposeTensor(t.data);
  }

  applyOp(opName: string, ...tensors: Tensor[]){
      // let op = this.opRegistry.get(opName);
  }

  add(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.add(tensorA.data, tensorB.data);
    return this.createTensor(outID, this.getTensorShape(outID), tensorA.dtype);
  }

  sub(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.sub(tensorA.data, tensorB.data);
    return this.createTensor(outID, this.getTensorShape(outID), tensorA.dtype);
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
    return this.createTensor(outID, this.getTensorShape(outID), tensorA.dtype);
  }
  div(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    if(tensorA.dtype !== DType.float32 || tensorB.dtype !== DType.float32){
      throw new Error(`oprands in div should has dtype float32`);
    }
    let outID = this.engine.div(tensorA.data, tensorB.data);
    return this.createTensor(outID, this.getTensorShape(outID), DType.float32);
  }


  matmul(a: Tensor, b: Tensor, transposeA: boolean = false, transposeB: boolean = false): Tensor{
    let outID = this.engine.matmul(a.data, b.data, transposeA, transposeB);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  sin(a: Tensor): Tensor{
    let outID = this.engine.sin(a.data);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  eq(a: Tensor | number, b: Tensor | number): Tensor{
    let tensorA: Tensor = (a instanceof Tensor) ? a : this.tensor([a]);
    let tensorB: Tensor = (b instanceof Tensor) ? b : this.tensor([b]);
    let outID = this.engine.eq(tensorA.data, tensorB.data);
    return this.createTensor(outID, this.getTensorShape(outID), tensorA.dtype);
  }

  log(a: Tensor): Tensor{
    let outID = this.engine.log(a.data);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);;
  }

  cos(a: Tensor): Tensor{
    let outID = this.engine.cos(a.data);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
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
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  exp(a: Tensor): Tensor{
    let outID = this.engine.exp(a.data);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  reshape(a: Tensor, shape: number[]){
      let unspec_axis: number = undefined;
      shape.forEach((value, index) => {
        if(value < 0){
            if(value !== -1){
                throw Error(`expect shape[${index}] >= -1, got ${value}`);
            }
            if(unspec_axis){
                throw Error(`multiple -1 in shape: ${unspec_axis} and ${index}`);
            }
            unspec_axis = index;
        }
      });

      let outID = this.engine.reshape(a.data, shape, (unspec_axis === undefined) ? -1 : unspec_axis);
      return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  removeAxis(a: Tensor, axis: number[]){
    let ax = axis.map((value, index) => {
        return value - index;
    });
    let outID = this.engine.removeAxis(a.data, ax);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
  }

  addAxis(a: Tensor, axis: number[]){
    let outID = this.engine.addAxis(a.data, axis);
    return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
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
            return this.createTensor(outID, this.getTensorShape(outID), a.dtype);
        }
        else{
            let outID = this.engine.reduce(a.data, mode, axis);
            let out = this.createTensor(outID, this.getTensorShape(outID), a.dtype);
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

  index_one_hot(t: Tensor, index: Tensor, axis: number = 1, keepdims: boolean = false){
    let outID = this.engine.index_one_hot(t.data, index.data, axis);
    let out = this.createTensor(outID, this.getTensorShape(outID), t.dtype);
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

  astype(t:Tensor, dtype: DType): Tensor{
      if(t.dtype === dtype){
          return t;
      }
      else{
        let outID = this.engine.astype(t.data, dtype);
        return this.createTensor(outID, this.getTensorShape(outID), dtype);
      }
  }

  argmax(t: Tensor, axis: null | number, keepdims: boolean = false){
    if(axis === null){
        if(keepdims){
            throw Error("can not set axis=null and keepdims=true");
        }
        let flatten = this.flattern(t);
        let outID = this.engine.argmax(flatten.data, 0);
        return this.createTensor(outID, this.getTensorShape(outID), t.dtype);
    }
    else{
        let outID = this.engine.argmax(t.data, axis);
        // output dtype must be int32 not t.dtype
        let out = this.createTensor(outID, this.getTensorShape(outID), DType.int32);
        if(keepdims){
            return out;
        }
        return this.squeeze(out, axis);
    }
  }

  startScope(name?: string) {
    const scopeInfo: ScopeState = {
      track: [],
      name: 'unnamed scope',
      id: this.nextScopeId++
    };
    this.scopeStack.push(scopeInfo);
    this.activeScope = scopeInfo;
  }

  endScope(result?: Tensor) {
    // console.log("exit scope: ", this.activeScope.id, this.activeScope.track);
    // Dispose the arrays tracked in this scope.
    for (let i = 0; i < this.activeScope.track.length; i++) {
      const tensor = this.activeScope.track[i];
      if (!result || tensor.data !== result.data) {
        // console.log("Disposing of ", tensor);
        ENGINE.disposeTensor(tensor); // tensor.dispose();
      }
    }
    
    const oldScope = this.scopeStack.pop();
    this.activeScope = this.scopeStack.length === 0 ?
        null :
        this.scopeStack[this.scopeStack.length - 1];
    if(result && result.scopeid == oldScope.id){
        this.track(result);
    }
  }


  tidy(fn: Function): Tensor{
    let result: Tensor;
    return this.scopedRun(
        () => this.startScope(), () => this.endScope(result), () => {
            result = fn();
            return result;
        });
  }

  scopedRun(start: () => void, end: () => void, f: () => Tensor): Tensor {
    start();
    try {
      const res = f();
      end();
      return res;
    } catch (ex) {
      end();
      throw ex;
    }
  }

  cleanup(){
      // console.log(this.tensorMap);
      if(this.engine){
        this.engine.delete();
      }
  }
}


let globalNameSpace : {megGlobals: Map<string, any>};

export function getGlobalNameSpace() : {megGlobals: Map<string, any>}{
  if (globalNameSpace == null) {
    // tslint:disable-next-line:no-any
    let ns: any;
    if (typeof (window) !== 'undefined') {
        ns = window;
    } else if (typeof (global) !== 'undefined') {
        ns = global;
    } else if (typeof (process) !== 'undefined') {
        ns = process;
    } else if (typeof (self) !== 'undefined') {
        ns = self
    } else {
        throw new Error('Could not find a global object');
    }
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
