import { isEntityName } from "typescript";
import { DType, TypedArray } from "./dtypes";
import {ENGINE} from './engine';

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

  

  print() {

  }
}
