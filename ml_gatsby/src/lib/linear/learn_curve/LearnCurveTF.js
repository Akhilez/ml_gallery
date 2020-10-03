import * as tf from "@tensorflow/tfjs"

export class LearnCurveTF {
  constructor(state) {
    this.state = state
    this.order = state.order
    this.model = this.createModel(this.order)
    this.data = this.getInitialData()
  }

  createModel(size) {
    const model = tf.sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [size] }))
    model.compile({
      optimizer: "adam",
      loss: "meanSquaredError",
      metrics: ["accuracy"],
    })
    return model
  }

  getWeights() {
    return this.model.getWeights()[0].squeeze().arraySync()
  }

  getInitialData() {
    const x = tf.randomUniform([50], -1, 1)
    const w = tf.tensor([-1.6, 2.3, 2.6, -1.2, -0.7])
    const order = w.shape[0]

    const new_x = []

    for (let xi of x) {
      const new_xi = []
      for (const j in w) {
        new_xi.push(xi.pow(order - j))
      }
      new_x.push(new_xi)
    }

    console.log(new_x)
  }
}
