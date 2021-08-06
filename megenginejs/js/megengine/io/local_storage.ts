import { DType, TypedArray } from "../dtypes";
import { ENGINE } from "../engine";
import { Tensor } from "../tensor";
import {inferSizeFromShape, getByteSizeFromDtype} from "../utils";
import {WeightHandler, StateDict, TensorSpec, MgeModel} from "./index";
import {concatenateTypedArrays, arrayBufferToBase64String, base64StringToArrayBuffer} from "./io_utils";

export class LocalStorageHandler implements WeightHandler{
  model_path: string

  constructor(model_path: string){
    this.model_path = model_path
  }

  save(state_dict: StateDict): void{
    let specs: TensorSpec[] = []
    let typed_arrays: TypedArray[] = []
    state_dict.forEach((tensor, key) => {
      specs.push({
        name: key,
        shape: tensor.shape,
        dtype: tensor.dtype
      });
      typed_arrays.push(ENGINE.readSync(tensor));
    });
    let artifact: MgeModel = {
      specs: specs,
      weights: arrayBufferToBase64String(concatenateTypedArrays(typed_arrays))
    }
    window.localStorage.setItem(`${this.model_path}.mge`, JSON.stringify(artifact));
  }

  load(): StateDict{
    let artifact = window.localStorage.getItem(`${this.model_path}.mge`);
    let {specs, weights}: MgeModel = JSON.parse(artifact);
    let array_buffer = base64StringToArrayBuffer(weights);
    let offset = 0;
    let state_dict = new Map();
    let value: TypedArray;
    specs.forEach(spec => {
      let data_length = inferSizeFromShape(spec.shape) * getByteSizeFromDtype(spec.dtype);
      let buffer = array_buffer.slice(offset, offset + data_length);
      if(spec.dtype === DType.float32){
        value = new Float32Array(buffer);
      }
      else if(spec.dtype === DType.int32){
        value = new Int32Array(buffer);
      }
      else if(spec.dtype === DType.int8){
        value = new Int8Array(buffer);
      }
      else if(spec.dtype === DType.uint8){
        value = new Uint8Array(buffer);
      } else {
        throw new Error(`Unsupported dtype in weight '${spec.name}': ${spec.dtype}`);
      }
      let tensor = ENGINE.tensor(value, {shape: spec.shape, dtype: spec.dtype});
      state_dict.set(spec.name, tensor);
    });
    return state_dict;
  }
}
