import * as tf from "@tensorflow/tfjs";
import * as tfvis from '@tensorflow/tfjs-vis';


export default class MnistClassifier {
    constructor(component) {
        this.component = component;
        this.model = null;

        this.initialize_model();
    }

    initialize_model() {

    }

    async captureP5Image(pixels) {
        let image = tf.scalar(255).sub(tf.tensor(Array.from(pixels))).div(255).reshape([280, 280, 4]);
        let [r, g, b, a] = image.split(4, 2);
        image = tf.stack([r, g, b]).reshape([280, 280, 3]);
        console.log(image.shape);

        console.log(image);

        let surface = tfvis.visor().surface({name: "Captured Image", tab: "capturedImage"});
        const canvas = document.createElement('canvas');
        canvas.width = 28;
        canvas.height = 28;
        canvas.style = 'margin: 4px;';
        await tf.browser.toPixels(image, canvas);
        surface.drawArea.appendChild(canvas);
        image.dispose();

    }

}