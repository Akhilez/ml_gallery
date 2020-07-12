const path = require("path");
const webpack = require('webpack');
const BundleTracker = require('webpack-bundle-tracker');
const {CleanWebpackPlugin} = require('clean-webpack-plugin');

module.exports = {
    context: __dirname,

    entry: {
        app: './static/app',
    },

    output: {
        path: path.resolve('./static/bundles/'),
        filename: "[name].bundle.js",
    },

    mode: 'development',

    plugins: [
        new CleanWebpackPlugin(),
        new BundleTracker({filename: './webpack-stats.json'}),
    ],

    module: {
        rules: [
            {
                test: /\.js$/,
                include: [path.resolve(__dirname, 'static'),],
                exclude: /node_modules/,
                use: ['babel-loader']
            }
        ]
    },

    resolve: {
        extensions: ['*', '.js', '.jsx']
    }

};