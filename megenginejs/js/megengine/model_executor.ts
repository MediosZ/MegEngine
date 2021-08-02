import { RecursiveArray, TypedArray, isTypedArray, DType} from "./dtypes";
import {ENGINE} from "./engine";
import {WasmModelExecutor} from "./wasm_dtypes";
import {TensorInfo} from './tensor';
import {inferShape, flatten} from './utils';

async function createFile(path: string): Promise<string> {
    return new Promise(async (resolve, reject) => {
        const response = await fetch(path);
        if(!response.ok){
            reject(`An error has occured: ${response.status}`);
        }
        const data = new Uint8Array(await response.arrayBuffer());
        const file_path = path.split("/").slice(-1)[0];
        ENGINE.wasm.FS_createDataFile('/', file_path, data, true, true, true);
        resolve(file_path);
    });
}

export async function createModelExecutor(url: string): Promise<ModelExecutor>{
    let path = await createFile(url);
    return new ModelExecutor(path);
}

function arrayEqual(array1: number[], array2: number[]): boolean{
    return array1.length === array2.length && array1.every(function(value, index) { return value === array2[index]})
}

function isShapeMismatch(input: number[], target: number[]){
    return !(arrayEqual(target, input) 
        ||arrayEqual(target.slice(1), input.slice(1))
    );
}

class ModelExecutor {
    model: WasmModelExecutor;

    constructor(path: string) {
        this.model = new ENGINE.wasm.WasmModelExecutor(path);
    }

    getInputShape(){
        const shapeString = this.model.getInputShape();
        return shapeString
            .substring(1, shapeString.length-1)
            .split(",")
            .map((x: string) => parseInt(x));
    }

    forward(data: RecursiveArray<number> | TypedArray, info :TensorInfo = {dtype: DType.float32}){
        let {shape, dtype} = info;
        let id;
        let inferedShape;
        let inputShape = this.getInputShape();
        if(isTypedArray(data)){
          inferedShape = shape || [(data as TypedArray).length];
          if(isShapeMismatch(inferedShape, inputShape)){
            throw Error(`input shape mismatch, expect: [${inputShape}] , but get [${inferedShape}]`);
          }
          const shapeBytes = new Int32Array(inferedShape);
          id = this.model.forward(shapeBytes, data, dtype);
        }
        else{
          inferedShape = shape || inferShape(data);
          if(isShapeMismatch(inferedShape, inputShape)){
            throw Error(`input shape mismatch, expect: [${inputShape}] , but get [${inferedShape}]`);
          }
          const shapeBytes = new Int32Array(inferedShape);
          let arrayBuffer = new Float32Array(flatten(data));
          id = this.model.forward(shapeBytes, arrayBuffer, dtype);
        }
        return ENGINE.createTensor(id, inferedShape, dtype);

    }

    delete() {
        this.model.delete();
    }
}