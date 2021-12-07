import typescript from '@rollup/plugin-typescript';
import commonjs from '@rollup/plugin-commonjs';
import resolve from '@rollup/plugin-node-resolve';
import node from '@rollup/plugin-node-resolve';
import {terser} from 'rollup-plugin-terser';
import nodePolyfills from 'rollup-plugin-node-polyfills';

export default {
  input: 'js/megengine/index.ts',
  output: {
    dir: 'dist',
    format: 'umd',
    name: 'mge',
    extend: true,
    sourcemap: false,
    globals: {
        'fs': 'fs',
        'path': 'path',
        'worker_threads': 'worker_threads',
        'perf_hooks': 'perf_hooks',
        'os': 'os'
    },
  },
  external: [
    'fs',
    'path',
    'worker_threads',
    'perf_hooks',
    'os'
  ],
  plugins: [typescript(), resolve(), node({preferBuiltins: false}),
    // Polyfill require() from dependencies.
    commonjs({
      ignore: ['crypto', 'node-fetch', 'util', 'fs', 'path', 'worker_threads', 'perf_hooks', 'os'],
      include: ['node_modules/**', 'wasm-out/**']
    }),
    terser({output: {preamble: "//wasm-sample", comments: false}, compress: {typeofs: false}}),
    nodePolyfills()
  ],
};
