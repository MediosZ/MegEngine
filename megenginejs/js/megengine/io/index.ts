import { DType } from "../dtypes";
import {StateDict} from "../modules/module";
export {StateDict} from "../modules/module";

export interface WeightHandler{
  save(state_dict: StateDict): void
  load(): StateDict
}

export interface TensorSpec{
  name: string
  shape: number[]
  dtype: DType
}

export interface MgeModel {
  specs: TensorSpec[]
  weights: string
}
