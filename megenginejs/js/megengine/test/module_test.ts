import * as mge from "../";

class Lenet extends mge.nn.Module{
  conv1: any;
  relu: any;
  pool: any;
  conv2: any;
  fc1: any;
  fc2: any;
  classifier: any;
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
  forward(inp: any){
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

describe("Module ",  function() {
  it("lenet ", function() {
    expect(async () => {
      await mge.run(()=>{
        let net = new Lenet();
        let inp = mge.rand([1, 1, 28, 28]);
        let out = net.forward(inp);
        expect(out.shape).toEqual([1, 10]);
      })
    });
  });
});