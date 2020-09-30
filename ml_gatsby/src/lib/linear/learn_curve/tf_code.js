import * as tf from "@tensorflow/tfjs"

export class LearnCurveTF {
  constructor(component) {
    this.order = component.state.order
    this.setState = component.setState
    this.model = this.createModel(this.order)
    this.getWeights()
  }

  createModel(size) {
    const model = tf.Sequential()
    model.add(tf.layers.dense({ units: 1, inputShape: [size] }))
    model.compile({ optimizer: "adam", loss: "mse", metrics: ["accuracy"] })
    return model
  }

  getWeights() {
    console.log(this.model)
  }
}
