import {DType, RecursiveArray} from './dtypes';

export function inferSizeFromShape(shape: number[]): number{
  if(shape.length == 0){
      return 0;
  }
  return shape.reduce((acc, cur)=>{
      return acc * cur
  }, 1)
}

export function flatten<T extends number|boolean|string|Promise<number>>(
  arr: T|RecursiveArray<T>, result: T[] = []): T[] {
if (result == null) {
  result = [];
}
if (Array.isArray(arr)) {
  for (let i = 0; i < arr.length; ++i) {
    flatten(arr[i], result);
  }
} else {
  result.push(arr as T);
}
return result;
}

export function inferShape(data: RecursiveArray<number>): number[]{
  let firstElem = data;
  let shape = [];
  while (Array.isArray(firstElem)) {
      shape.push(firstElem.length);
      firstElem = firstElem[0];
  }
  return shape
}

export function getByteSizeFromDtype(dtype: DType){
  if(dtype === DType.float32 || dtype === DType.int32){
    return 4;
  }
  else if(dtype === DType.int8 || dtype === DType.uint8){
    return 1;
  }
  else{
    throw new Error("unsupported dtype");
  }
}

export function calculateStrides(shape: number[]): number[]{
  // calculate strides of a tensor
  let strides = [1]
  for(let i = 0; i < shape.length - 1; i++){
    strides.push(strides[i] * shape[shape.length - i - 1])
  }
  return strides.reverse();
}
