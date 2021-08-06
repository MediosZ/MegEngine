import {WeightHandler, StateDict} from "./index";


class FileHandler implements WeightHandler{
  model_path: string

  constructor(model_path: string){
    this.model_path = model_path
  }
  save(state_dict: StateDict): void{
    /*
    const url = window.URL.createObjectURL(new Blob(
      [], {type: 'application/octet-stream'}));
    const anchor = document.createElement('a');
    anchor.download = `${this.model_path}.mge`;
    anchor.href = url;
    () => anchor.dispatchEvent(new MouseEvent('click'));
    */
  }

  load(): StateDict{
    return new Map();
  }
}