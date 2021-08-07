
import { DType, TypedArray} from "../dtypes";
import { ENGINE } from "../engine";
import {StateDict} from "../modules/module";
import {inferSizeFromShape, getByteSizeFromDtype} from "../utils";
import {concatenateTypedArrays, arrayBufferToBase64String, base64StringToArrayBuffer} from "./io_utils";

export interface TensorSpec{
  name: string
  shape: number[]
  dtype: DType
}

export interface MgeModel {
  specs: TensorSpec[]
  weights: string
}
export class WeightHandler{
  encodeArtifact(state_dict: StateDict): MgeModel {
    let specs: TensorSpec[] = []
    let typed_arrays: TypedArray[] = []
    state_dict.forEach((tensor, key) => {
      specs.push({
        name: key,
        shape: tensor.shape,
        dtype: tensor.dtype
      });
      let arr = ENGINE.readSync(tensor);
      typed_arrays.push(arr);
    });
    let artifact: MgeModel = {
      specs: specs,
      weights: arrayBufferToBase64String(concatenateTypedArrays(typed_arrays))
    }
    return artifact;
  }
  decodeArtifact(artifact: string): StateDict {
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
      offset += data_length;
      let tensor = ENGINE.tensor(value, {shape: spec.shape, dtype: spec.dtype});
      state_dict.set(spec.name, tensor);
    });
    return state_dict;
  }
}