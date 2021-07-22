import {ENGINE} from "./index";

describe("Engine init",  function() {
  jasmine.DEFAULT_TIMEOUT_INTERVAL = 1000000;
  it("test output ", async function() {
    await ENGINE.init();
    let random_tensor = ENGINE.rand([100, 1],0,0.01)
    expect(random_tensor.shape).toEqual([100, 1]);
  });
});
