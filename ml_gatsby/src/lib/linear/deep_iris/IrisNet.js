import * as tf from "@tensorflow/tfjs"
import iris from "src/data/iris/iris_train.json"

export default class IrisNet {
  constructor(component) {
    this.component = component

    const [x, y] = this.getTrainingData()
    this.x = x
    this.y = y

    this.initialize_net()

    this.is_training = false
  }

  initialize_net() {
    this.net = tf.sequential()

    if (this.component.state.nNeurons.length <= 0)
      this.net.add(
        tf.layers.dense({ units: 3, inputShape: [4], activation: "softmax" })
      )
    else {
      this.net.add(
        tf.layers.dense({
          units: this.component.state.nNeurons[0],
          inputShape: [4],
          activation: "relu",
        })
      )
      this.component.state.nNeurons.forEach((nNeurons, index) => {
        if (index > 0) {
          this.net.add(tf.layers.dense({ units: nNeurons, activation: "relu" }))
        }
      })
      this.net.add(tf.layers.dense({ units: 3, activation: "softmax" }))
    }
    this.net.compile({
      loss: "categoricalCrossentropy",
      optimizer: "adam",
    })
  }

  train() {
    const onEpochEnd = (epoch, logs) => {
      this.component.setState({
        lossData: this.component.state.lossData.concat([
          {
            index: this.component.state.lossData.length,
            loss: logs.loss,
          },
        ]),
      })
      if (!this.component.state.isTraining) this.net.stopTraining = true
    }

    this.net.fit(this.x, this.y, {
      epochs: 10000,
      batchSize: 4,
      callbacks: { onEpochEnd },
    })
  }

  predict(x) {
    x = tf.tensor(x).div(100)
    return this.net.predict(x).squeeze().arraySync()
  }

  getTrainingData() {
    const x = this.normalize(
      iris.map(item => [
        item.sepal_length,
        item.sepal_width,
        item.petal_length,
        item.petal_width,
      ])
    )

    const y = tf.tensor(
      iris.map(item => [
        item.species === "setosa" ? 1 : 0,
        item.species === "versicolor" ? 1 : 0,
        item.species === "virginica" ? 1 : 0,
      ])
    )

    return [x, y]
  }

  normalize(x) {
    // x is list of lists of size 4
    x = tf.tensor(x)
    const mins = tf.tensor([4.5, 2.5, 1, 0])
    const maxes = tf.tensor([2.1, 1, 4.6, 2.1])
    return x.sub(mins).div(maxes)
  }
}
