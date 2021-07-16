
import wasmPath from "megenginejs/meg.wasm";

import {
    ENGINE, setWasmPath, GradManager, SGD,
    Conv2D, MaxPool2D, Linear, Module, RELU, CrossEntropy} from "megenginejs";

class Lenet extends Module{
    constructor(){
        super();
        this.conv1 = Conv2D(1, 6, 5);
        this.relu = RELU();
        this.pool = MaxPool2D(2);
        this.conv2 = Conv2D(6, 16, 5);
        this.fc1 = Linear(16 * 5 * 5, 120);
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
    let lenet = new Lenet();
    let images = ENGINE.rand([6, 1, 32, 32], 0, 1);

    let gm = new GradManager().attach(lenet.parameters());
    let opt = SGD(lenet.parameters(), 0.001);

    for(let i = 0; i < 5; i++){
      console.log(`epoch ${i}`);
      gm.backward(() => {
        let loss =  CrossEntropy(lenet.forward(images), [1,3,4,2,4,3])
        ENGINE.printTensor(loss, "Loss is ");
        return loss;
      })
      opt.step();
    }

    ENGINE.cleanup();
}

run();
