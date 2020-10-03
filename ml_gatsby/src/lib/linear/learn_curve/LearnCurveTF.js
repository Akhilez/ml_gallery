import * as tf from "@tensorflow/tfjs"

export class LearnCurveTF {
  constructor() {
    this.order = 0
    this.data = this.getInitialData()
    this.interrupt = null
  }

  createModel(size) {
    const model = tf.sequential()
    model.add(
      tf.layers.dense({
        units: 1,
        inputShape: [size + 1],
        useBias: false,
        kernelInitializer: "zeros",
        biasInitializer: "zeros",
      })
    )
    model.compile({
      optimizer: "adam",
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    })
    return model
  }

  train(epochs = 300) {
    this.interrupt = null
    this.model.fit(this.data[0], this.data[1], {
      batchSize: 4,
      epochs: epochs,
      callbacks: {
        onEpochEnd: epoch => {
          if (this.interrupt) {
            this.model.stopTraining = true
            this.interrupt = null
          }
        },
      },
    })
  }

  stopTraining() {
    this.interrupt = "stopped"
  }

  setOrder(order) {
    this.order = order
    this.data[0] = tf.tensor(
      this.getHigherOrderInputs(this.data[2].arraySync(), order)
    )
    this.model = this.createModel(order)
  }

  predict(x) {
    x = this.getHigherOrderInputs(x, this.order)
    x = tf.tensor(x)
    return this.model.predict(x).squeeze().arraySync()
  }

  getWeights() {
    return this.model.getWeights()[0].squeeze().dataSync()
  }

  getData() {
    return this.data.map(data => data.arraySync())
  }

  getInitialData() {
    const w = tf.tensor([-0.85, -1.6, 2.3, 2.6, -1.2, -0.7])
    const order = w.shape[0] - 1

    const x_old = tf.randomUniform([50], -1, 1)
    const x = tf.tensor(this.getHigherOrderInputs(x_old.dataSync(), order))

    const y = x.dot(w)

    return [x, y, x_old]
  }

  getHigherOrderInputs(x, order) {
    // x: list of floats
    const new_x = []
    for (let xi of x) {
      const new_xi = []
      for (let j = 0; j < order + 1; j++) new_xi.push(Math.pow(xi, order - j))
      new_x.push(new_xi)
    }
    return new_x
  }
}
