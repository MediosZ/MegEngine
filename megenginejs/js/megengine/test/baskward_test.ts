import * as mge from "../";

describe("Backward test", function() {
  it("add backward ", async function() {
    await mge.run( async () => {
      let a = mge.tensor([[1,2,3],[4,5,6]]);
      let b = mge.tensor([[4,5,6],[1,2,3]]);
      let gm = new mge.GradManager();
      gm.attach([a, b]);
      gm.backward(() => {
        return a.add(b);
      })

      expect(mge.equal(a.grad, [[1,1,1],[1,1,1]])).toBe(true);
      expect(mge.equal(b.grad, [[1,1,1],[1,1,1]])).toBe(true);
    });
  });

  it("mul backward ", async function() {
    await mge.run( async () => {
      let a = mge.tensor([[1,2,3],[4,5,6]]);
      let b = mge.tensor([[4,5,6],[1,2,3]]);
      let gm = new mge.GradManager();
      gm.attach([a, b]);
      gm.backward(() => {
        return a.mul(b);
      })
      expect(mge.equal(a.grad, b)).toBe(true);
      expect(mge.equal(b.grad, a)).toBe(true);
    });
  });
});
