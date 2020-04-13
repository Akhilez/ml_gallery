/*
import mnist_test from '../data/mnist/mnist_handwritten_test';
import mnist_train from '../data/mnist/mnist_handwritten_train';
import * as tf from '@tensorflow/tfjs';


export class MLHelper {

    constructor () {
        this.weights1 = tf.randomUniform([150, 784], -0.5, 0.5, "float32");
        this.weights2 = tf.randomUniform([10, 150], -0.5, 0.5, "float32");
    }

    createModel() {
        this.model = tf.sequential();
        this.model.add(tf.layers.dense({inputShape: [784], units: 150, useBias: true}));
        this.model.add(tf.layers.dense({units: 10, useBias: true, activation: 'sigmoid'}))
        this.model.compile({optimizer: 'sgd', loss: 'cros'});
        const history = this.model.fit([], [], {batchSize: 100, epochs: 10});
    }

    createModel2() {
        this.generator = tf.sequential();
        this.generator.add(tf.layers.dense({inputShape: [100], units: 150, useBias: true}));
        this.generator.add(tf.layers.dense({units: 400, useBias: true}));
        this.generator.add(tf.layers.dense({units: 784, useBias: true, activation: 'sigmoid'}));

        this.discriminator = tf.sequential();
        this.discriminator.add(tf.layers.dense({inputShape: [784], units: 400, useBias: true}));
        this.discriminator.add(tf.layers.dense({units: 150, useBias: true}));
        this.discriminator.add(tf.layers.dense({units: 100, useBias: true}));
        this.discriminator.add(tf.layers.dense({units: 1, useBias: true, activation: 'sigmoid'}));
    }

    train (epochs) {
        // Preproccessing
        for (let data of mnist_test) {
            let x = tf.tensor(data.image);
            let y = tf.oneHot(data.label, 10);

            let h1 = tf.dot(x, this.weights1.transpose());
            console.log(h1);
            return;
        }

    }
}


*/