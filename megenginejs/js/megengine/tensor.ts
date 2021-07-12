import { DType, TypedArray } from "./dtypes";

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

  print() {

  }
}
