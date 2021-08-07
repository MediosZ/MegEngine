import * as mge from "megenginejs";

async function run() {
    console.log("Linear Regression!");
    mge.run(async () => {
      const epoch = 10;
      const num_inputs = 2;
      const num_examples = 300;
      const true_w = mge.tensor([[2], [-3.4]]);
      const true_b = mge.tensor([4.2]);
      let features = mge.rand([num_examples, num_inputs])
      let labels = mge.tidy(() => {
          return features.matmul(true_w)
          .add(true_b)
          .add(mge.rand([num_examples, 1],0,0.01));
      });
      
      let linear = mge.nn.Linear(2, 1);
  
      let gm = new mge.GradManager();
      // gm.attach([w, b]);
      gm.attach(linear.parameters())
  
      let learning_rate = 0.5;
      let opt = mge.optim.SGD(linear.parameters(), learning_rate);
  
      for(let i = 0; i < epoch; i++){
        gm.backward(() => {
          // let output = features.matmul(w).add(b);
          return mge.tidy(() => {
              let output = linear.forward(features);
              let loss = mge.loss.MSE(output, labels);
              return loss;
          })
        })
        opt.step();
      }
      console.log(`true_w: ${true_w.toString()}, \n get ${linear.weight.toString()}`);
      console.log(`true_b: ${true_b.toString()}, \n get ${linear.bias.toString()}`);
    });
}

run();
