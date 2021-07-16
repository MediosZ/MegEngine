
import wasmPath from "megenginejs/meg.wasm";

import {ENGINE, setWasmPath, GradManager, SGD, MSE, Linear} from "megenginejs";

async function run() {
    setWasmPath(wasmPath);
    console.log("Linear Regression!");

    await ENGINE.init();
    const epoch = 10;
    const num_inputs = 2;
    const num_examples = 300;
    const true_w = ENGINE.tensor([[2], [-3.4]]);
    const true_b = ENGINE.tensor([4.2]);
    let features = ENGINE.rand([num_examples, num_inputs])

    let labels = features
        .matmul(true_w)
        .add(true_b)
        .add(ENGINE.rand([num_examples, 1],0,0.01));
    let linear = Linear(2, 1);
/*
    let w = ENGINE.rand([num_inputs, 1], 0, 0.005);
    let b = ENGINE.tensor([0]);
*/
    let gm = new GradManager();
    // gm.attach([w, b]);
    gm.attach(linear.parameters())

    let learning_rate = 0.5;
    let opt = SGD(linear.parameters(), learning_rate);
    // let opt = SGD([w, b], learning_rate);

    for(let i = 0; i < epoch; i++){
      gm.backward(() => {
        // let output = features.matmul(w).add(b);
        let output = linear.forward(features);
        let loss = MSE(output, labels);
        ENGINE.printTensor(loss, "Loss: ");
        return loss;
      })
      opt.step();
    }
    ENGINE.printTensor(linear.weight, `w [${ENGINE.readSync(true_w)}] : `);
    ENGINE.printTensor(linear.bias, `b [${ENGINE.readSync(true_b)}] : `);
    // ENGINE.printTensor(w, `w [${ENGINE.readSync(true_w)}] : `)
    // ENGINE.printTensor(b, `w [${ENGINE.readSync(true_b)}] : `)

    ENGINE.cleanup();
}

run();
