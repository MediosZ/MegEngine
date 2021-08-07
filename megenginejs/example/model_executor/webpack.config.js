const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development',
  entry: {
    app: './src/index.js',
  },
  devServer: {
    contentBase: './dist',
    headers: {
        "Access-Control-Allow-Origin": "*",
        "Cross-Origin-Opener-Policy": "same-origin",
        "Cross-Origin-Embedder-Policy": "require-corp",
    },
  },
  optimization:{
    minimize: false,
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'Model Executor',
    }),
  ],
  module: {
    rules: [
      {
        test: /\.wasm$/i,
        type: 'javascript/auto',
        use: [
          {
            loader: 'file-loader',
          },
        ],
      },
      {
        test: /\.(txt|mge)$/i,
        use: [
          {
            loader: 'file-loader',
          },
        ],
      },
    ],
  }
};
