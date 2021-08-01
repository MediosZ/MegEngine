const fs = require('fs');

const BASE_PATH = './wasm-out/';
const WORKER_PATH = `${BASE_PATH}meg.worker.js`;

const workerContents = fs.readFileSync(WORKER_PATH, "utf8");
fs.chmodSync(WORKER_PATH, 0o644);
fs.writeFileSync(`${WORKER_PATH}`,
  `export const wasmWorkerContents = "${workerContents.replace(/"/g, "").replace(/\n/g, "")}";`);