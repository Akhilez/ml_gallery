import * as tf from "@tensorflow/tfjs";

export default class MLHelper {

    constructor(){
        this.m = tf.randomUniform([1], -0.5, 0.5);
        this.c = tf.randomUniform([1], -0.5, 0.5);
        this.lr = 0.1;
    }

    fullPass (x, y) {
        x = tf.tensor([x]);
        y = tf.tensor([y]);

        let f = (m, c) => y.sub(m.mul(x).add(c)).square();
        let g = tf.grads(f);

        let loss = f(this.m, this.c);

        let [dm, dc] = g([this.m, this.c]);
        this.m = this.m.sub(dm.mul(this.lr));
        this.c = this.c.sub(dc.mul(this.lr));

        return loss;
    }

    getWeights(){
        return {
            m: Array.from(this.m.dataSync())[0],
            c: Array.from(this.c.dataSync())[0]
        }
    }


    generateRandomLineData(m, c, count) {
        let randomX = tf.randomUniform([count], 0, 1).dataSync();
        let randomY = [...randomX].map((x) => {
            return (parseFloat(m) * x) + parseFloat(c);
        });
        return {x: Array.from(randomX), y: randomY};
    }


}