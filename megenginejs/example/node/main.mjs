import {ENGINE, GradManager, SGD, MSE, Linear} from "megenginejs";

async function run() {
    console.log("Linear Regression!");

    await ENGINE.init();
    const epoch = 10;
    const num_inputs = 2;
    const num_examples = 300;
    const true_w = ENGINE.tensor([[2], [-3.4]]);
    const true_b = ENGINE.tensor([4.2]);
    let features = ENGINE.rand([num_examples, num_inputs])
    let labels = ENGINE.tidy(() => {
        return features.matmul(true_w)
        .add(true_b)
        .add(ENGINE.rand([num_examples, 1],0,0.01));
    });
    
    let linear = Linear(2, 1);

    let gm = new GradManager();
    gm.attach(linear.parameters())

    let learning_rate = 0.5;
    let opt = SGD(linear.parameters(), learning_rate);

    for(let i = 0; i < epoch; i++){
      gm.backward(() => {
        return ENGINE.tidy(() => {
            let output = linear.forward(features);
            let loss = MSE(output, labels);
            ENGINE.printTensor(loss, "Loss: ");
            return loss;
        })
      })
      opt.step();
    }
    ENGINE.printTensor(linear.weight, `w [${ENGINE.readSync(true_w)}] : `);
    ENGINE.printTensor(linear.bias, `b [${ENGINE.readSync(true_b)}] : `);
    ENGINE.cleanup();
}

run();
