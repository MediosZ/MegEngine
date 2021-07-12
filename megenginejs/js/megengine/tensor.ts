import { DType, TypedArray } from "./dtypes";
import {ENGINE} from './engine';

export class Tensor{
  data: number;
  grad?: Tensor;
  dtype: DType;
  shape: number[];
  offset: number;

  constructor(data:number, shape: number[], offset: number,
      dtype: DType = DType.float32){
      this.data = data
      this.dtype = dtype
      this.shape = shape
      this.offset = offset
  }

  add(b: Tensor): Tensor{
    return ENGINE.add(this, b);
  }
  add_(b: Tensor){
    ENGINE.add_(this, b);
  }
  sub(b: Tensor): Tensor{
    return ENGINE.sub(this, b);
  }
  sub_(b: Tensor){
    ENGINE.sub_(this, b);
  }
  mul(b: Tensor): Tensor{
    return ENGINE.mul(this, b);
  }
  matmul(b: Tensor): Tensor{
    return ENGINE.matmul(this, b);
  }
  div(b: Tensor): Tensor{
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

  

  print() {

  }
}
