export enum DType{
  float32,
  int32,
  int8,
  uint8
}

export declare type TypedArray = Float32Array | Int32Array | Uint8Array | Int8Array;

export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}
