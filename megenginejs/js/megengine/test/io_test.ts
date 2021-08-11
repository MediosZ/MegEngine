import * as mge from "../";

describe("IO test", function() {
  it("read ", async function() {
    await mge.run( async () => {
      let t = mge.tensor([[1,2,3],[4,5,6]]);
      expect(mge.read(t)).toEqual(new Float32Array([1,2,3,4,5,6]));
      
    });
  });
});
