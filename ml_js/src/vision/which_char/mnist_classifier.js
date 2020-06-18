import * as tf from "@tensorflow/tfjs";


export default class MnistClassifier {
    constructor(component) {
        this.component = component;
        this.model = null;

        this.initialize_model();
    }

    initialize_model() {

    }
}