import * as mge from "../";

describe("Generate Random Tensor",  function() {
  it("random shape", async function() {
    mge.run( () => {
      let random_tensor = mge.rand([100, 1],0,0.01)
      expect(random_tensor.shape).toEqual([100, 1]);
    });
  });

  it("ones", async function() {
    await mge.run( async () => {
      let a = mge.ones([3,2]);
      expect(mge.equal(
        a,
        mge.tensor([[1,1], [1,1], [1,1]]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);
    });
  });
  it("zeros", async function() {
    await mge.run( async () => {
      let a = mge.zeros([3,2]);
      expect(mge.equal(
        a,
        mge.tensor([[0,0], [0,0], [0,0]]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);
    });
  });
});
