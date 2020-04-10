import * as tf from "@tensorflow/tfjs";

export default class MLHelper {
    generateRandomLineData(m, c, count) {
        let randomX = tf.randomUniform([count], 0, 1);
        let randomY = Array.from(randomX.dataSync()).map((x)=>m*x + c);
        return {x: Array.from(randomX.dataSync()), y: randomY};
    }
}