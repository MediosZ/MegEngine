module.exports = function(config) {
  config.set({
      basePath: '',
      frameworks: ["jasmine", "karma-typescript"],
      files: [
          // Serve the wasm file as a static resource.
          {pattern: 'wasm-out/*.wasm', included: false},
          // Import the generated js library from emscripten.
          { pattern: 'wasm-out/*.js' },
          { pattern: "js/megengine/**/*.ts" },
          // { pattern: "test/**/*.ts" }
      ],

      preprocessors: {
          "js/megengine/**/*.ts": ["karma-typescript"],
          "test/**/*.ts": ["karma-typescript"],
          // ! let karma find generated js
          'wasm-out/**/*.js': ['karma-typescript']
      },

      reporters: ["progress", "karma-typescript"],

      browsers: ["Chrome"], // "Safari", 
      karmaTypescriptConfig: {
        tsconfig: 'tsconfig.test.json',
        compilerOptions: {allowJs: true, declaration: false},
        bundlerOptions: {
          sourceMap: true,
          transforms: [
            require('karma-typescript-es6-transform')({
              presets: [
                // ensure we get es5 by adding IE 11 as a target
                ['@babel/env', {'targets': {'ie': '11'}, 'loose': true}]
              ]
            }),
          ]
        },
        include: ['js/megengine/', 'wasm-out/']
      },
      proxies: {
        '/base/node_modules/karma-typescript/dist/client/meg.wasm':
            '/base/wasm-out/meg.wasm'
      }
  });
};
