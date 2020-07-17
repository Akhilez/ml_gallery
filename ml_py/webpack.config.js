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
            },
            {
                test: /\.(png|svg|jpg|gif|jpeg)$/,
                use: [
                    'file-loader',
                ],
            },
            {
                test: /\.(json5|json)$/i,
                loader: 'json5-loader',
                type: 'javascript/auto',
            },
            {
                test: /\.css$/i,
                use: ['style-loader', 'css-loader'],
            },
        ]
    },

    resolve: {
        extensions: ['*', '.js', '.jsx']
    }

};