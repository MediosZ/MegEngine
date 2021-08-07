import {StateDict } from "../modules/module";
import {WeightHandler} from "./weight_handler";

export class LocalStorageHandler extends WeightHandler{
  model_path: string

  constructor(model_path: string){
    super()
    this.model_path = model_path
  }

  save(state_dict: StateDict){
    let artifact = this.encodeArtifact(state_dict);
    window.localStorage.setItem(`${this.model_path}.mge`, JSON.stringify(artifact));
  }

  load(): StateDict{
    let artifact = window.localStorage.getItem(`${this.model_path}.mge`);
    return this.decodeArtifact(artifact);
  }
}
