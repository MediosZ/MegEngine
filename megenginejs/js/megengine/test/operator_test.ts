import * as mge from "../";

describe("Operator Test", function() {
  it("equal ", async function() {
    await mge.run( async () => {
      expect(mge.equal(1, 1)).toBe(true);
      expect(mge.equal([1,2,3,4], [1,2,3,4])).toBe(true);
      expect(mge.equal(mge.tensor([1,2,3,4]), [1,2,3,4])).toBe(true);
      expect(mge.equal(mge.tensor([1,2,3,4]), mge.tensor([1,2,3,4]))).toBe(true);
      expect(mge.equal(mge.tensor([[1,2],[3,4]]), [[1,2],[3,4]])).toBe(true);
      expect(mge.equal(mge.tensor([1,2,3,4]), mge.tensor([1,2,4]))).toBe(false);
      
    });
  });

  it("add ", async function() {
    await mge.run( async () => {
      let a = mge.add(mge.tensor([1,2,3,4]), mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([2,4,6,8]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,2,3,8]))).toBe(false);
    });
  });

  it("add_ ", async function() {
    await mge.run( async () => {
      let a = mge.tensor([1,2,3,4])
      mge.add_(a, mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([2,4,6,8]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,2,3,8]))).toBe(false);
    });
  });

  it("sub_ ", async function() {
    await mge.run( async () => {
      let a = mge.tensor([4,3,2,1])
      mge.sub_(a, mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([3,1,-1,-3]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);;
    });
  });

  it("sub ", async function() {
    await mge.run( async () => {
      let a = mge.sub(mge.tensor([4,3,2,1]), mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([3,1,-1,-3]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);
    });
  });

  it("mul ", async function() {
    await mge.run( async () => {
      let a = mge.mul(mge.tensor([4,3,2,1]), mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([4,6,6,4]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);
    });
  });

  it("div ", async function() {
    await mge.run( async () => {
      let a = mge.div(mge.tensor([4,3,2,1]), mge.tensor([1,2,3,4]));
      expect(mge.equal(
        a,
        mge.tensor([4, 1.5, 2/3, 0.25]))).toBe(true);
      expect(mge.equal(
        a,
        mge.tensor([2,1,3,3]))).toBe(false);
    });
  });

  it("log ", async function() {
    await mge.run( async () => {
      let a = mge.log(mge.tensor([4,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([Math.log(4), Math.log(3), Math.log(2), Math.log(1)]))).toBe(true);
    });
  });


  it("relu ", async function() {
    await mge.run( async () => {
      let a = mge.relu(mge.tensor([-1,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([0,3,2,1]))).toBe(true);
    });
  });

  it("exp ", async function() {
    await mge.run( async () => {
      let a = mge.exp(mge.tensor([4,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([Math.exp(4), Math.exp(3), Math.exp(2), Math.exp(1)]))).toBe(true);
    });
  });

  it("square ", async function() {
    await mge.run( async () => {
      let a = mge.square(mge.tensor([4,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([16, 9, 4, 1]))).toBe(true);
    });
  });

  it("cos ", async function() {
    await mge.run( async () => {
      let a = mge.cos(mge.tensor([4,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([Math.cos(4), Math.cos(3), Math.cos(2), Math.cos(1)]))).toBe(true);
    });
  });

  it("sin ", async function() {
    await mge.run( async () => {
      let a = mge.sin(mge.tensor([4,3,2,1]));
      expect(mge.equal(
        a,
        mge.tensor([Math.sin(4), Math.sin(3), Math.sin(2), Math.sin(1)]))).toBe(true);
    });
  });

  it("matmul ", async function() {
    await mge.run( async () => {
      let a = mge.matmul(mge.tensor([4,3,2,1]), mge.tensor([[1],[2],[3],[4]]));
      expect(mge.equal(
        a,
        20)).toBe(true);
      mge.matmul(mge.rand([4,3]), mge.rand([3,4]));
      mge.matmul(mge.rand([4]), mge.rand([4,3]));
      mge.matmul(mge.rand([4,3]), mge.rand([3]));
      mge.matmul(mge.rand([4,4,3]), mge.rand([4,3,4]));
      mge.matmul(mge.rand([3,4,3]), mge.rand([3,4]));
      mge.matmul(mge.rand([3,3,4,4]), mge.rand([3,3,4,4]));
    });
  });

  it("equal ", async function() {
    await mge.run( async () => {
      let a = mge.eq(mge.tensor([4,3,2,1]), mge.tensor([1,3,2,4]));
      expect(mge.equal(
        a,
        [0,1,1,0])).toBe(true);
    });
  });

  it("mean ", async function() {
    await mge.run( async () => {
      let a = mge.mean(mge.tensor([[4,3],[2,1]]));
      expect(mge.equal(
        a,
        [2.5])).toBe(true);
      let b = mge.mean(mge.tensor([[4,3],[2,1]]), 0, false);
      expect(mge.equal(
        b,
        [3, 2])).toBe(true);
      let c = mge.mean(mge.tensor([[4,3],[2,1]]), 1, false);
      expect(mge.equal(
        c,
        [3.5, 1.5])).toBe(true);
      let d = mge.mean(mge.tensor([[4,3],[2,1]]), 1, true);
      expect(mge.equal(
        d,
        [[3.5], [1.5]])).toBe(true);
    });
  });


  it("sum ", async function() {
    await mge.run( async () => {
      let a = mge.sum(mge.tensor([[4,3],[2,1]]));
      expect(mge.equal(
        a,
        [10])).toBe(true);
      let b = mge.sum(mge.tensor([[4,3],[2,1]]), 0, false);
      expect(mge.equal(
        b,
        [6, 4])).toBe(true);
      let c = mge.sum(mge.tensor([[4,3],[2,1]]), 1, false);
      expect(mge.equal(
        c,
        [7, 3])).toBe(true);
      let d = mge.sum(mge.tensor([[4,3],[2,1]]), 1, true);
      expect(mge.equal(
        d,
        [[7], [3]])).toBe(true);
    });
  });


  it("max ", async function() {
    await mge.run( async () => {
      let a = mge.max(mge.tensor([[4,3],[2,1]]));
      expect(mge.equal(
        a,
        [4])).toBe(true);
      let b = mge.max(mge.tensor([[4,3],[2,1]]), 0, false);
      expect(mge.equal(
        b,
        [4, 3])).toBe(true);
      let c = mge.max(mge.tensor([[4,3],[2,1]]), 1, false);
      expect(mge.equal(
        c,
        [4, 2])).toBe(true);
      let d = mge.max(mge.tensor([[4,3],[2,1]]), 1, true);
      expect(mge.equal(
        d,
        [[4], [2]])).toBe(true);
    });
  });


  it("min ", async function() {
    await mge.run( async () => {
      let a = mge.min(mge.tensor([[4,3],[2,1]]));
      expect(mge.equal(
        a,
        [1])).toBe(true);
      let b = mge.min(mge.tensor([[4,3],[2,1]]), 0, false);
      expect(mge.equal(
        b,
        [2, 1])).toBe(true);
      let c = mge.min(mge.tensor([[4,3],[2,1]]), 1, false);
      expect(mge.equal(
        c,
        [3, 1])).toBe(true);
      let d = mge.min(mge.tensor([[4,3],[2,1]]), 1, true);
      expect(mge.equal(
        d,
        [[3], [1]])).toBe(true);
    });
  });

  it("argmax ", async function() {
    await mge.run( async () => {
      let a = mge.argmax(mge.tensor([[4,3,2,1], [2,4,3,1], [3,2,1,4]]));
      expect(mge.equal(
        a,
        mge.astype(mge.tensor([0,1,1,2]), mge.DType.int32))).toBe(true);
      let b = mge.argmax(mge.tensor([[4,3,2,1], [2,4,3,1], [3,2,1,4]]), 1);
      expect(mge.equal(
        b,
        mge.astype(mge.tensor([0,1,3]), mge.DType.int32))).toBe(true);
      let c = mge.argmax(mge.tensor([[4,3,2,1], [2,4,3,1], [3,2,1,4]]), 1, true);
      expect(mge.equal(
        c,
        mge.astype(mge.tensor([[0],[1],[3]]), mge.DType.int32))).toBe(true);
    });
  });

  it("astype ", async function() {
    await mge.run( async () => {
      let a = mge.tensor([[4,3,2,1], [2,4,3,1], [3,2,1,4]]);
      expect(a.dtype).toBe(mge.DType.float32);
      let b = mge.astype(a, mge.DType.int32);
      expect(b.dtype).toBe(mge.DType.int32);
      let c = mge.astype(a, mge.DType.int8);
      expect(c.dtype).toBe(mge.DType.int8);
      let d = mge.astype(a, mge.DType.uint8);
      expect(d.dtype).toBe(mge.DType.uint8);
    });
  });



});
