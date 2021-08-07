import { DType, RecursiveArray, TypedArray } from "../dtypes";
import {ENGINE} from '../engine';
import {Tensor, TensorInfo} from "../tensor";

export function tensor(data: RecursiveArray<number> | TypedArray, info :TensorInfo = {dtype: DType.float32}): Tensor{
  return ENGINE.tensor(data, info);
}

export function rand(shape: number[], mean: number = 0.0, std: number = 1.0, dtype: DType = DType.float32): Tensor{
  return ENGINE.rand(shape, mean, std, dtype);
}

export function zeros(shape: number[], dtype: DType = DType.float32): Tensor{
  return ENGINE.zeros(shape, dtype);
}

export function ones(shape: number[], dtype: DType = DType.float32): Tensor{
  return ENGINE.ones(shape, dtype);
}

export function read(tensor: Tensor):TypedArray {
  return ENGINE.readSync(tensor);
}

export function add(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.add(a, b);
}

export function sub(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.sub(a, b);
}

export function sub_(a: Tensor | number, b: Tensor | number){
  return ENGINE.sub_(a, b);
}

export function add_(a: Tensor | number, b: Tensor | number){
  return ENGINE.add_(a, b);
}

export function mul(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.mul(a, b);
}

export function div(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.div(a, b);
}

export function matmul(a: Tensor, b: Tensor,transposeA: boolean = false, transposeB: boolean = false): Tensor{
  return ENGINE.matmul(a, b, transposeA, transposeB);
}

export function eq(a: Tensor | number, b: Tensor | number): Tensor{
  return ENGINE.eq(a, b);
}

export function log(a: Tensor): Tensor{
  return ENGINE.log(a);
}

export function mean(a: Tensor, axis?: number, keepdims=false): Tensor{
  return ENGINE.mean(a, axis, keepdims);
}

export function sum(a: Tensor, axis?: number, keepdims=false): Tensor{
  return ENGINE.sum(a, axis, keepdims);
}

export function max(a: Tensor, axis?: number, keepdims=false): Tensor{
  return ENGINE.max(a, axis, keepdims);
}

export function min(a: Tensor, axis?: number, keepdims=false): Tensor{
  return ENGINE.min(a, axis, keepdims);
}

export function square(a: Tensor): Tensor{
  return ENGINE.square(a);
}

export function relu(a: Tensor): Tensor{
  return ENGINE.relu(a);
}

export function exp(a: Tensor): Tensor{
  return ENGINE.exp(a);
}

export function squeeze(a: Tensor, axis?: number): Tensor{
  return ENGINE.squeeze(a, axis);
}

export function unsqueeze(a: Tensor, axis?: number): Tensor{
  return ENGINE.unsqueeze(a, axis);
}

export function flattern(a: Tensor): Tensor{
  return ENGINE.flattern(a);
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
