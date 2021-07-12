// src/main.js

import {ENGINE, GradManager} from './index';

async function main() {
    await ENGINE.init();
    // console.log(ENGINE.wasm);
    // ENGINE.testBackward();
    const epoch = 10;
    const num_inputs = 2;
    const num_examples = 500;
    const true_w = ENGINE.tensor([[2], [-3.4]]);
    const true_b = ENGINE.tensor([4.2]);
    let features = ENGINE.rand([num_examples, num_inputs])

    let labels = features
        .matmul(true_w)
        .add(true_b)
        .add(ENGINE.rand([num_examples],0,0.01));
    // console.log(ENGINE.readSync(labels.id));

    let w = ENGINE.rand([num_inputs, 1], 0, 0.005);
    let b = ENGINE.tensor([0]);

    let gm = new GradManager();
    let bs = ENGINE.tensor([num_examples]);
    let lr = ENGINE.tensor([0.001]);

    gm.attach([w, b]);

    for(let i = 0; i < epoch; i++){
      gm.backward(() => {
        let output = features.matmul(w).add(b);
        let loss = output.sub(labels).square().sum();
        
        ENGINE.printTensor(loss, "Loss: ");
        return loss;
      })
      gm.optimize(lr, bs);
    }
    ENGINE.printTensor(w, `w [${ENGINE.readSync(true_w)}] : `)
    ENGINE.printTensor(b, `w [${ENGINE.readSync(true_b)}] : `)

    // ENGINE.apply();
    /*
    let r = ENGINE.rand([3,4]);
    console.log(ENGINE.readSync(r.id));
    */
    ENGINE.cleanup();


}

main();
