# MegEngine.js

MegEngine.js 构建在 MegEngine 之上，通过将MegEngine编译成WebAssembly，使之可以在Web上运行。

## 使用

可以通过两种方式使用MegEngine.js，通过NPM或者<script>标签。

### 通过NPM

使用·npm·或者·yarn·安装megengine.js即可使用，更多用例参考examples文件夹。

### 通过 <script> 标签

通过在html文件中直接包含megengine.js来使用，如下：

```html

<script src="https://cdn.jsdelivr.net/npm/megenginejs"></script>

<script>
  // some code
</script>

```

## 开发

首先安装MegEngine中的说明配置好系统环境，然后执行以下操作：

```bash 
cd megenginejs
# 安装依赖
yarn
# 编译wasm
bash ./scripts/build-wasm.sh
# 编译megengine.js
yarn build
```

## 测试

编译megengine.js完成之后，执行 `yarn test`进行测试。

