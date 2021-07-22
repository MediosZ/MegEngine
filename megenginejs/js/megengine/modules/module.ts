import {Tensor, Parameter} from "../tensor";

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
}