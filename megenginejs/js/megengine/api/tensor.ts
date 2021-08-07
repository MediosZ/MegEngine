import { DType, RecursiveArray, TypedArray } from "../dtypes";
import {ENGINE} from '../engine';
import {Tensor, TensorInfo} from "../tensor";

export function tensor(data: RecursiveArray<number> | TypedArray, info :TensorInfo = {dtype: DType.float32}): Tensor{
  return ENGINE.tensor(data, info);
}

export function rand(shape: number[], mean: number = 0.0, std: number = 1.0, dtype: DType = DType.float32): Tensor{
  return ENGINE.rand(shape, mean, std, dtype);
}

export function astype(tensor: Tensor, dtype: DType): Tensor{
  return ENGINE.astype(tensor, dtype);
}

export function argmax(tensor: Tensor, axis: number = 0,keepdims: boolean = false): Tensor{
  return ENGINE.argmax(tensor, axis, keepdims);
}

export function reshape(tensor: Tensor, shape: number[]): Tensor{
  return ENGINE.reshape(tensor, shape);
}

export function eq(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.eq(a, b);
}