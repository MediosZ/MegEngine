
import wasmPath from "megenginejs/meg.wasm";

import {
    ENGINE, setWasmPath, GradManager, SGD,
    Conv2D, MaxPool2D, Linear, Module, RELU, CrossEntropy, DType} from "megenginejs";

import { MnistData } from "./mnist";

class SubNet extends Module{
    constructor(){
        super();
        this.c1 = Conv2D(1,6,5);
        this.fc = Linear(120, 20);
    }
}

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
        return ENGINE.tidy(() => {
            let x = this.relu.forward(this.pool.forward(this.conv1.forward(inp)));
            x = this.relu.forward(this.pool.forward(this.conv2.forward(x)));
            x = ENGINE.reshape(x, [x.shape[0], -1]);
            x = this.relu.forward(this.fc1.forward(x));
            x = this.relu.forward(this.fc2.forward(x));
            return this.classifier.forward(x);
        });
    }
}

async function mnist() {
    console.log("Running Megenginejs");
    setWasmPath(wasmPath);
    await ENGINE.init();
    
    let batch_size = 500;
    let mnistData = new MnistData(batch_size);
    await mnistData.load();
    let lenet = new Lenet();
    let gm = new GradManager().attach(lenet.parameters());
    let opt = SGD(lenet.parameters(), 0.5);
    for(let i = 0; i < 1; i++){
      console.log(`epoch ${i}`);
      let trainGen = mnistData.getTrainData();
      while(true){
          let {value, done} = trainGen.next();
          if(done){
              break;
          }
          let input = ENGINE.tidy(() => ENGINE.reshape(ENGINE.tensor(value["data"]), [batch_size, 1, 28, 28]));
          let label = ENGINE.tidy(() => ENGINE.argmax(ENGINE.astype(ENGINE.reshape(ENGINE.tensor(value["label"]), [batch_size, 10]), DType.int32), 1));
          let now = Date.now();
          gm.backward(() => {
            return ENGINE.tidy(() => {
                let out = lenet.forward(input);
                let loss =  CrossEntropy(out, label)
                ENGINE.printTensor(loss, "Loss is ");
                return loss;
            });
          })
          
          opt.step();
          ENGINE.disposeTensor(input);
          ENGINE.disposeTensor(label);
          let then = Date.now();
          console.log(`step ${i} ${then - now}`);
      }
    };
    
    ENGINE.cleanup();
    
}

async function run(){
    console.log("Running Megenginejs");
    setWasmPath(wasmPath);
    await ENGINE.init();
    let lenet = new Lenet();
    let dict = lenet.state_dict();
    console.log(dict);
    console.log("save lenet");
    ENGINE.cleanup();
}

mnist();
