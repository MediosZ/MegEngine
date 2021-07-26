import { TypedArray } from "../dtypes";
import { ENGINE } from "../engine";
import {Tensor, Parameter} from "../tensor";


function extractProperties<T>(obj: object, predicate: (value: any) => boolean): [string, T][] {
    return Object.keys(obj)
        .map(key => [key, obj[key as keyof object]])
        .filter(([key, value]) => predicate(value)) as [string, T][];
}

const extractTensors:(obj: object) => [string, Tensor][] = (obj: object) => extractProperties(obj, (value: any) => value instanceof Tensor);
const extractModules:(obj: object) => [string, Module][] = (obj: object) => extractProperties(obj, (value: any) => value instanceof Module);

class StateDict extends Map<string, Tensor>{

}

export class Module{
    forward(inp: Tensor){
        throw Error("not implemented");
    }

    parameters(){
        let res: Tensor[] = [];
        let retrive = (obj: Module, res: Tensor[]) => {
            let varibales = Object.keys(obj)
                .map(key => obj[key as keyof Module])
                .filter(x => x instanceof Parameter || x instanceof Module);
            for(let v of varibales){
                if(v instanceof Tensor){
                    res.push(v);
                }
                else if(v instanceof Module){
                    retrive(v, res);
                }
            }
        }
        retrive(this, res);
        return res;
    }

    _state_dict(): StateDict{
        let res: StateDict = new StateDict();
        let retrive = (obj: Module, res: StateDict, _prefix: string = "") => {
            let prefix = _prefix === "" ? "" : _prefix + ".";
            let tensors = extractTensors(obj);
            for(let item of tensors){
                res.set(prefix + item[0], item[1]);
            }
            let modules = extractModules(obj);
            for(let item of modules){
                retrive(item[1], res, prefix + item[0]);
            }
        }
        retrive(this, res);
        return res;
    }

    state_dict(){
        return Object.assign({}, 
            ...Object.entries(this._state_dict()).map(
                ([k, v]) => ({[k]: ENGINE.readSync(v)})
            ));
    }

    load_state_dict(_state_dict: {[key: string]: TypedArray}){
        let state_dict = this._state_dict()
        Object.keys(state_dict).map(function(key, index) {
            let oldTensor = state_dict.get(key);
            const shapeBytes = new Int32Array(oldTensor.shape);
            ENGINE.engine.replaceTensor(oldTensor.data, shapeBytes, _state_dict[key], oldTensor.dtype);
        });
    }
}