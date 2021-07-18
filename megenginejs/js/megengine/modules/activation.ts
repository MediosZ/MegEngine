import {ENGINE} from "../engine";
import {Tensor} from "../tensor";
import {Module} from "./module";

class _RELU extends Module{
    constructor(){
        super();
    }

    forward(inp: Tensor){
        let outID = ENGINE.engine.relu(inp.data);
        let offset = ENGINE.getMemOffset(outID);
        let out = new Tensor(outID, ENGINE.getTensorShape(outID), offset);
        ENGINE.track(out);
        ENGINE.tensorMap.set(outID, out);
        return out;
    }
}


export function RELU() {
    return new _RELU();
}
