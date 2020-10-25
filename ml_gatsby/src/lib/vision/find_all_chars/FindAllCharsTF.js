import * as tf from "@tensorflow/tfjs"

export class FindAllCharsTF {
  constructor(component) {
    this.component = component
  }

  pixToTensor(pixels) {
    return tf
      .scalar(255)
      .sub(tf.tensor(Array.from(pixels)))
      .div(255)
      .reshape([448, 448, 4])
      .split(4, 2)[0]
      .resizeBilinear([112, 112])
      .reshape([-1, 1, 112, 112])
  }

  predict(pixels) {
    let image = this.pixToTensor(pixels)

    // TODO: Call the predict api with img data
  }

  async initialize_model() {
    // TODO: Make the connection
  }

  initialize_data() {}
}
