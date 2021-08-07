const HtmlWebpackPlugin = require('html-webpack-plugin');

module.exports = {
  mode: 'development',
  entry: {
    app: './src/index.js',
  },
  devServer: {
    contentBase: './dist',
  },
  plugins: [
    new HtmlWebpackPlugin({
      title: 'Playground',
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
    ],
  }
};
