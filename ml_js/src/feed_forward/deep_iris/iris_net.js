import * as tf from "@tensorflow/tfjs";
import iris from "../../data/iris/iris_train.json"


export default class IrisNet {
    constructor(component) {
        this.component = component;
        this.trainingData = tf.tensor2d(iris.map(item => [
            item.sepal_length, item.sepal_width, item.petal_length, item.petal_width,
        ]));
        this.outputData = tf.tensor2d(iris.map(item => [
            item.species === "setosa" ? 1 : 0,
            item.species === "virginica" ? 1 : 0,
            item.species === "versicolor" ? 1 : 0,
        ]));
        this.initialize_net();

        this.is_training = false;

    }

    initialize_net() {
        this.net = tf.sequential();

        if (this.component.state.nNeurons.length <= 0)
            this.net.add(tf.layers.dense({units: 3, inputShape: [4], activation: 'softmax'}));
        else {
            this.net.add(tf.layers.dense({
                units: this.component.state.nNeurons[0],
                inputShape: [4],
                activation: 'sigmoid'
            }));
            this.component.state.nNeurons.forEach((nNeurons, index) => {
                if (index > 0) {
                    this.net.add(tf.layers.dense({units: nNeurons, activation: 'relu'}));
                }
            });
            this.net.add(tf.layers.dense({units: 3, activation: "softmax"}));
        }
        this.net.compile({
            loss: "meanSquaredError",
            optimizer: tf.train.adam(.001),
        })
    }

    async train() {
        let self = this;

        function onEpochEnd(epoch, logs) {
            self.component.setState({
                lossData: self.component.state.lossData.concat([{
                    index: self.component.state.lossData.length,
                    loss: logs.loss
                }])
            });
            if (!self.component.state.isTraining) {
                self.net.stopTraining = true;
            }
        }

        let history = await this.net.fit(this.trainingData, this.outputData, {
            epochs: 500, batchSize: 4,
            callbacks: {
                onEpochEnd
            }
        });

        if (this.component.state.isTraining) this.train();
    }

    predict(x) {
        return this.net.predict(x)
    }

}
