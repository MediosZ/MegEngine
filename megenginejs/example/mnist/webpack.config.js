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
    port: 3000, 
    public: 'localhost:3000',
  },
  optimization:{
    minimize: false,
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'Mnist',
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
        test: /\.mge$/i,
        use: [
          {
            loader: 'file-loader',
          },
        ],
      },
    ],
  }
};
