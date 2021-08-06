import {WeightHandler} from "./index";
import {StateDict } from "../modules/module";


export class FileHandler extends WeightHandler{
  model_path: string

  constructor(model_path: string){
    super()
    this.model_path = model_path
  }
  save(state_dict: StateDict): void{
    let artifact = this.encodeArtifact(state_dict);
    const url = window.URL.createObjectURL(new Blob(
      [JSON.stringify(artifact)],
      {type: 'application/json'}));
    const anchor = document.createElement('a');
    anchor.download = `${this.model_path}.mge`;
    anchor.href = url;
    anchor.dispatchEvent(new MouseEvent('click'));
  }

  async load(path: string): Promise<StateDict>{
    return new Promise(async (resolve, reject) => {
      let response = await fetch(path);
      if (!response.ok) {
        const message = `File at ${path} not found`;
        throw new Error(message);
      }
      let artifact = await response.text();
      resolve(this.decodeArtifact(artifact));
    });
  }
}