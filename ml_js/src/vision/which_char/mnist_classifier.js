import * as tf from "@tensorflow/tfjs";
import * as tfvis from '@tensorflow/tfjs-vis';
// import model_json from './models/model.json';


export default class MnistClassifier {
    constructor(component) {
        this.component = component;
        this.model = null;

        this.initialize_model();
    }

    async initialize_model() {
        this.model = await tf.loadGraphModel('./models/model.json');
    }

    captureP5Image(pixels) {
        let image = tf.scalar(255).sub(tf.tensor(Array.from(pixels))).div(255).reshape([280, 280, 4]);
        image = image.split(4, 2)[0];
        image = image.resizeBilinear([28, 28]);

        let output = this.model.predict(image);
        console.log(output);

    }

    async plotImage(image) {
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