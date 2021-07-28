export enum DType{
  float32,
  int32,
  int8,
  uint8
}
export function isTypedArray(x: any): boolean{
    return (x instanceof Float32Array) || (x instanceof Float64Array) || (x instanceof Int32Array) || (x instanceof Uint8Array);
}

export declare type TypedArray = Float32Array | Int32Array | Uint8Array | Int8Array;

export interface RecursiveArray<T extends any> {
  [index: number]: T|RecursiveArray<T>;
}
