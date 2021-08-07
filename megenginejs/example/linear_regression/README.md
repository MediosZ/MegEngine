# 线性回归

为了让Webpack能打包WASM文件，我们需要在代码中手动引入WASM文件，具体如下：


Then we obtain the final serving path of the WASM binaries that were shipped on
NPM, and use `setWasmPaths` to let the library know the serving locations:

```ts
import wasmPath from "megenginejs/meg.wasm";

import {setWasmPath} from "megenginejs";

setWasmPath(wasmPath);
```
