import * as mge from "../";

describe("Engine Initialize",  function() {
  it("success ", function() {
    expect(async () => {
      await mge.run(()=>{})
    });
  });
});

describe("NextFrame",  function() {
  it("success ", function() {
    expect(async () => {
      await mge.run(async ()=>{
        await mge.nextFrame();
      })
    });
  });
});

