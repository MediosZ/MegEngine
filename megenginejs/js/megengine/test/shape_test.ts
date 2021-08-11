import * as mge from "../";

describe("shape test", function() {
  it("squeeze ", async function() {
    await mge.run( async () => {
      let t = mge.rand([3,1,2,3]);
      let a = mge.squeeze(t, 1);
      expect(a.shape).toEqual([3,2,3]);
      expect(() => {
        mge.squeeze(t, 2)
      }).toThrowError("axis 2 should be 1, got 2");
    });
  });

  it("unsqueeze ", async function() {
    await mge.run( async () => {
      let t = mge.rand([3,2,3]);
      let a = mge.unsqueeze(t, 1);
      expect(a.shape).toEqual([3,1,2,3]);
      let b = mge.unsqueeze(t);
      expect(b.shape).toEqual([3,2,3,1]);
    });
  });

  it("flattern ", async function() {
    await mge.run( async () => {
      let t = mge.rand([3,2,3]);
      let a = mge.flattern(t);
      expect(a.shape).toEqual([18]);
    });
  });

  it("reshape ", async function() {
    await mge.run( async () => {
      let t = mge.rand([3,2,3]);
      let a = mge.reshape(t, [2, 9]);
      expect(a.shape).toEqual([2, 9]);
      let b = mge.reshape(t, [2, -1]);
      expect(b.shape).toEqual([2, 9]);
      let c = mge.reshape(t, [-1, 3]);
      expect(c.shape).toEqual([6, 3]);
      expect(() => {
        mge.reshape(t, [3,3,3])
      }).toThrowError("the shape of tensor mismatch, expect 27, get 18");
      expect(() => {
        mge.reshape(t, [3,3,-2])
      }).toThrowError("expect shape[2] >= -1, got -2");
      expect(() => {
        mge.reshape(t, [3,-1,-1])
      }).toThrowError("multiple -1 in shape at 1 and 2");
    });
  });
});
