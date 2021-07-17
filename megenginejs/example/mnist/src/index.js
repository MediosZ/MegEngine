
import wasmPath from "megenginejs/meg.wasm";

import {
    ENGINE, setWasmPath, GradManager, SGD,
    Conv2D, MaxPool2D, Linear, Module, RELU, CrossEntropy, DType} from "megenginejs";

import { MnistData } from "./mnist";

class Lenet extends Module{
    constructor(){
        super();
        this.conv1 = Conv2D(1, 6, 5);
        this.relu = RELU();
        this.pool = MaxPool2D(2);
        this.conv2 = Conv2D(6, 16, 5);
        this.fc1 = Linear(16 * 4 * 4, 120);
        this.fc2 = Linear(120, 84);
        this.classifier = Linear(84, 10);
    }
    forward(inp){
        let x = this.relu.forward(this.pool.forward(this.conv1.forward(inp)));
        x = this.relu.forward(this.pool.forward(this.conv2.forward(x)));
        x = ENGINE.reshape(x, [x.shape[0], -1]);
        x = this.relu.forward(this.fc1.forward(x));
        x = this.relu.forward(this.fc2.forward(x));
        return this.classifier.forward(x);
    }
}

async function run() {
    setWasmPath(wasmPath);
    console.log("Running Megenginejs");

    await ENGINE.init();

    let arr = Int32Array.from([1,2,3,4,5,6]);
    console.log(arr);
    let ten = ENGINE.tensor(arr);
    console.log(ten);
    ENGINE.printTensor(ten)
    let res = ENGINE.reshape(ten, [2,3]);
    console.log(res);
    ENGINE.printTensor(res)

    /*
    let arr = [1,2,3, 4, 5, 6];
    console.log(arr);
    let ten = ENGINE.tensor(arr);
    console.log(ten);
    ENGINE.printTensor(ten)
    let res = ENGINE.reshape(ten, [2,3]);
    console.log(res);
    ENGINE.printTensor(res)
    */
    /*
    let mnistData = new MnistData(100);
    await mnistData.load();
    console.log("load mnist");
    
    let lenet = new Lenet();

    let gm = new GradManager().attach(lenet.parameters());
    let opt = SGD(lenet.parameters(), 0.001);

    for(let i = 0; i < 1; i++){
      console.log(`epoch ${i}`);
      let trainGen = mnistData.getTrainData();
      while(true){
          let {value, done} = trainGen.next();
          if(done){
              break;
          }
  
          let input = ENGINE.reshape(ENGINE.tensor(value["data"]), [100, 1, 28, 28]);
          console.log(input.shape)
          let label = ENGINE.tensor(value["label"])// ENGINE.reshape(, [100, 10]);
          console.log(label.shape)
          
          ENGINE.printTensor(label);
          gm.backward(() => {
            let out = lenet.forward(input);
            console.log(out.shape)
            let loss =  CrossEntropy(out, label)
            console.log("finish")
            //ENGINE.printTensor(loss, "Loss is ");
            return loss;
          })
          // opt.step();
          break;
      }
    };

    */
    ENGINE.cleanup();
}

run();
