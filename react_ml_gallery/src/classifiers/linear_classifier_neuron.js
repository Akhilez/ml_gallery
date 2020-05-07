import * as tf from "@tensorflow/tfjs";

export default class LinearClassifierNeuron {
    constructor() {
        this.data = this.getRandomX();

        this.w = tf.randomUniform([2], -0.1, 0.1);
        this.b = tf.randomUniform([1], -0.1, 0.1);

        this.lr = 0.5;
    }

    fullPass() {
        tf.util.shuffle(this.data);
        let x = this.data.slice([0, 0], [-1, 2]);
        let y = this.data.slice([0, 2]).reshape([-1]);

        let forward = (w, b) => tf.losses.sigmoidCrossEntropy(y, x.dot(w).add(b));

        let grad = tf.grads(forward);

        let loss = forward(this.w, this.b);

        let [dw, db] = grad([this.w, this.b]);

        this.w = this.w.sub(dw.mul(this.lr));
        this.b = this.b.sub(db.mul(this.lr));

        // loss.print();
        return loss;
    }

    getRandomX() {
        let size = 5;

        let points_a = tf.randomUniform([size, 2], 0, 0.5);
        let points_b = tf.randomUniform([size, 2], 0.5, 1);

        points_a = tf.concat([points_a, tf.ones([size, 1])], 1);
        points_b = tf.concat([points_b, tf.zeros([size, 1])], 1);

        return tf.concat([points_a, points_b], 0);
    }

    getDataPoints() {
        return this.data.arraySync();
    }

    getMC() {
        let w = this.w.dataSync();
        let c = this.b.dataSync();

        /*
        ax + by + c = 0.5
        by = 0.5 - ax - c
        y = (0.5/b) + (-a/b)x + (-c/b)
        y = (-a/b)x + ((-c+0.5)/b)
         */

        return {
            w: -w[0] / w[1],
            b: (-c + 0.5) / w[1]
        };
    }

}