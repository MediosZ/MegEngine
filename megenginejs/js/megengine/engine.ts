import {DType, RecursiveArray, TypedArray} from './dtypes';
import {inferShape, inferSizeFromShape, flatten} from './utils';
import {init, MegEngine, setWasmPath} from './backend';
import {Tensor} from './tensor';

export {setWasmPath} from './backend';

class Engine{
  tensorMap: Map<number, Tensor>
  wasm: MegEngine
  curId: number
  engine: object

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

  getMemOffset(id: number): number{
    return this.engine.getTensorOffset(id);
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

  add(a: Tensor, b: Tensor): Tensor{
    let outID = this.engine.add(a.data, b.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sub(a: Tensor, b: Tensor): Tensor{
    let outID = this.engine.sub(a.data, b.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sub_(a: Tensor, b: Tensor){
    this.engine.sub_(a.data, b.data);
  }

  add_(a: Tensor, b: Tensor){
    this.engine.add_(a.data, b.data);
  }

  mul(a: Tensor, b: Tensor): Tensor{
    let outID = this.engine.mul(a.data, b.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  div(a: Tensor, b: Tensor): Tensor{
    let outID = this.engine.div(a.data, b.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }


  matmul(a: Tensor, b: Tensor): Tensor{
    let outID = this.engine.matmul(a.data, b.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, [a.shape[0], b.shape[1]], offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  sin(a: Tensor): Tensor{
    let outID = this.engine.sin(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }

  cos(a: Tensor): Tensor{
    let outID = this.engine.cos(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, a.shape, offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  mean(a: Tensor): Tensor{
    let outID = this.engine.mean(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, [1], offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  max(a: Tensor): Tensor{
    let outID = this.engine.max(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, [1], offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  min(a: Tensor): Tensor{
    let outID = this.engine.min(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, [1], offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  sum(a: Tensor): Tensor{
    let outID = this.engine.sum(a.data);
    let offset = this.getMemOffset(outID);
    let out = new Tensor(outID, [1], offset, DType.float32);
    this.tensorMap.set(outID, out);
    return out;
  }
  square(a: Tensor): Tensor{
    return this.mul(a, a);
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
