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

    let labels =
    ENGINE.add(
      ENGINE.add(
        ENGINE.matmul(
          features,
          true_w ) ,
        true_b),
      ENGINE.rand(
        [num_examples],
        0,
        0.01)
    );
    // console.log(ENGINE.readSync(labels.id));

    let w = ENGINE.rand([num_inputs, 1], 0, 0.005);
    let b = ENGINE.tensor([0]);

    let gm = new GradManager();
    let bs = ENGINE.tensor([num_examples]);
    let lr = ENGINE.tensor([0.001]);
    ENGINE.printTensor(bs);
    ENGINE.printTensor(lr);
    gm.attach([w, b]);

    for(let i = 0; i < epoch; i++){
      gm.backward(() => {
        let output =
        ENGINE.add(
          ENGINE.matmul(
            features,
            w),
          b
        );

        let loss =
        ENGINE.sum(
          ENGINE.square(
            ENGINE.sub(
              output,
              labels))
        );
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
