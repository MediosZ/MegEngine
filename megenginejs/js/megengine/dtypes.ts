export enum DType{
  float32,
  int32
}

export declare type TypedArray = Float32Array | Float64Array | Int32Array | Uint8Array;

export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}
