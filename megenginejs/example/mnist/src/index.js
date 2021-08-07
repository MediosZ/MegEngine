import wasmPath from "megenginejs/meg.wasm";
import * as mge from "megenginejs";

import mnistWeight from "../mnist.mge";
import { MnistData } from "./mnist";

class Lenet extends mge.nn.Module{
    constructor(){
        super();
        this.conv1 = mge.nn.Conv2D(1, 6, 5);
        this.relu = mge.nn.RELU();
        this.pool = mge.nn.MaxPool2D(2);
        this.conv2 = mge.nn.Conv2D(6, 16, 5);
        this.fc1 = mge.nn.Linear(16 * 4 * 4, 120);
        this.fc2 = mge.nn.Linear(120, 84);
        this.classifier = mge.nn.Linear(84, 10);
    }
    forward(inp){
        return mge.tidy(() => {
            let x = this.relu.forward(this.pool.forward(this.conv1.forward(inp)));
            x = this.relu.forward(this.pool.forward(this.conv2.forward(x)));
            x = mge.reshape(x, [x.shape[0], -1]);
            x = this.relu.forward(this.fc1.forward(x));
            x = this.relu.forward(this.fc2.forward(x));
            return this.classifier.forward(x);
        });
    }
}

async function mnist() {
  try {
    console.log("Running Megmgejs");
    mge.setWasmPath(wasmPath);
    await mge.init();
    let handler = new mge.io.LocalStorageHandler("mnist");
    const batch_size = 500;
    const epoch = 1;
    let mnistData = new MnistData(batch_size);
    await mnistData.load();
    let lenet = new Lenet();

    let gm = new mge.GradManager().attach(lenet.parameters());
    let opt = mge.optim.SGD(lenet.parameters(), 0.3);
    for(let i = 0; i < epoch; i++){
      console.log(`epoch ${i}`);
      let trainGen = mnistData.getTrainData();
      while(true){
          let {value, done} = trainGen.next();
          if(done){
              break;
          }
          let input = mge.tidy(() => mge.reshape(mge.tensor(value["data"]), [batch_size, 1, 28, 28]));
          let label = mge.tidy(() => mge.argmax(mge.astype(mge.reshape(mge.tensor(value["label"]), [batch_size, 10]), mge.DType.int32), 1));
          let now = Date.now();
          gm.backward(() => {
            return mge.tidy(() => {
                let out = lenet.forward(input);
                let loss =  mge.loss.CrossEntropy(out, label)
                mge.printTensor(loss, "Loss is ");
                return loss;
            });
          })
          
          opt.step();
          mge.disposeTensor(input);
          mge.disposeTensor(label);
          let then = Date.now();
          console.log(`step ${i} ${then - now}`);
      }
      handler.save(lenet.state_dict());
      console.log("save weight");
    };
  }
  finally {
    mge.cleanup();
  }
}

async function mnistTest(){
  mge.setWasmPath(wasmPath);
  await mge.init();
  try{
    let batch_size = 500;
    let mnistData = new MnistData(batch_size);
    await mnistData.load();
  
    let lenet = new Lenet();
    // let handler = new LocalStorageHandler("mnist");
    let handler = new mge.io.FileHandler("mnist");
    lenet.load_state_dict(await handler.load(mnistWeight));
  
    let testGen = mnistData.getTestData();
    while(true){
        let {value, done} = testGen.next();
        if(done){
            break;
        }
        let input = mge.tidy(() => mge.reshape(mge.tensor(value["data"]), [batch_size, 1, 28, 28]));
        let label = mge.tidy(() => mge.argmax(mge.astype(mge.reshape(mge.tensor(value["label"]), [batch_size, 10]), mge.DType.int32), 1));
        
        let accTensor = mge.tidy(() => {
          let out = mge.argmax(lenet.forward(input), 1);
          return mge.eq(out, label).sum();
        });
        let acc = accTensor.item() / batch_size;
        console.log("accuracy is ", acc);
        mge.disposeTensor(input);
        mge.disposeTensor(label);
        mge.disposeTensor(accTensor);
    }
  }
  finally{
    mge.cleanup();
  }
}

mnist();