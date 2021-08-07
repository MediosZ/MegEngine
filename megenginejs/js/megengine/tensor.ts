import { DType} from "./dtypes";
import {ENGINE} from './engine';
import { calculateStrides } from "./utils";


export interface TensorInfo {
    shape?: number[];
    dtype: DType;
}

export class Parameter{}

export class Tensor extends Parameter{
  data: number;
  grad?: Tensor;
  dtype: DType;
  shape: number[];
  offset: number;
  requires_grad: boolean
  scopeid: number

  constructor(data:number, shape: number[], offset: number,
      dtype: DType = DType.float32){
      super();
      this.data = data
      this.dtype = dtype
      this.shape = shape
      this.offset = offset
      this.requires_grad = true
      this.scopeid = -1;
  }

  public get age() {
    return this.shape.length;
  }


  add(b: Tensor | number): Tensor{
    let that = this;
    return ENGINE.add(that, b);
  }
  add_(b: Tensor){
    ENGINE.add_(this, b);
  }
  sub(b: Tensor | number): Tensor{
    return ENGINE.sub(this, b);
  }
  sub_(b: Tensor){
    ENGINE.sub_(this, b);
  }
  mul(b: Tensor | number): Tensor{
    return ENGINE.mul(this, b);
  }
  matmul(b: Tensor): Tensor{
    return ENGINE.matmul(this, b);
  }
  div(b: Tensor | number): Tensor{
    return ENGINE.div(this, b);
  }

  sin(): Tensor{
    return ENGINE.sin(this);
  }
  cos(): Tensor{
    return ENGINE.cos(this);
  }

  min(): Tensor{
    return ENGINE.min(this);
  }
  max(): Tensor{
    return ENGINE.max(this);
  }
  sum(): Tensor{
    return ENGINE.sum(this);
  }

  mean(): Tensor{
    return ENGINE.mean(this);
  }

  square(): Tensor{
    return ENGINE.square(this);
  }

  slice(...args: string[]): Tensor{
      return ENGINE.zeros([1]);
  }

  item(): number {
    if(this.shape.length !== 1 ||this.shape[0] !== 1){
      throw new Error("only scalar can call item()");
    }
    return ENGINE.readSync(this)[0];
  }

  toString(): string{
    let data = ENGINE.readSync(this);
    const strides = calculateStrides(this.shape);

    const tensorToString = (shape: number[], pre: number[]):string => {
      if(shape.length == 1){
        let numRows = shape[0];
        const index = pre.map((value, i) => value * strides[i]).reduce((acc, cur) => acc + cur, 0);
        let result = [];
        for(let i = 0; i < numRows; i++){
          result.push(data[index + i].toFixed(5).toString());
        }
        return `[ ${result.join(", ")} ]`;
      }
      let numRows = shape[0];
      let result = [];
      for(let i = 0; i < numRows; i++){
        result.push(tensorToString(shape.slice(1), pre.concat([i])));
      }
      return shape.length > 2 ? `[ ${result.join(",\n\n")} ]` : `[ ${result.join(",\n")} ]`;
    }
    return `Tensor [${this.shape}] ${DType[this.dtype]} \n${tensorToString(this.shape, [])}`
  }

  print() {
    console.log(this.toString());
  }
}
