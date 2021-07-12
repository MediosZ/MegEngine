import typescript from '@rollup/plugin-typescript';
import html  from '@rollup/plugin-html';
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
    name: 'megjs',
    extend: true,
    sourcemap: true,
    globals: {
        'fs': 'fs',
        'path': 'path',
        'worker_threads': 'worker_threads',
        'perf_hooks': 'perf_hooks'
    },
  },
  external: [
    'fs',
    'path',
    'worker_threads',
    'perf_hooks',
  ],
  plugins: [typescript(), resolve(), node({preferBuiltins: false}),
    // Polyfill require() from dependencies.
    commonjs({
      ignore: ['crypto', 'node-fetch', 'util'],
      include: ['node_modules/**', 'wasm-out/**']
    }),
    html({
        fileName: "index.html"
    }),
    terser({output: {preamble: "//wasm-sample", comments: false}, compress: {typeofs: false}}),
    nodePolyfills()
  ],
};
