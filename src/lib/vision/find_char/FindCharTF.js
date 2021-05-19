import * as tf from "@tensorflow/tfjs"

const modelUrl =
  "https://storage.googleapis.com/akhilez/models/find_char/model.json"

const dataUrl =
  "https://storage.googleapis.com/akhilez/datasets/mnist/localization.json"

export class FindCharTF {
  constructor(component) {
    this.component = component
    this.model = null
    this.data = null
  }

  train() {}

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

    const [cxd, cyd, wd] = this.model.predict(image).dataSync()
    this.component.canvasRef.current.drawBoundingBox(cxd, cyd, wd)
  }

  async initialize_model() {
    this.model = await tf.loadLayersModel(modelUrl)

    if (false) {
      // I want to enable this later
      const lastLayer = this.model.layers[this.model.layers.length - 1]
      lastLayer.setWeights([
        tf.randomUniform([100, 10], 0.5, -0.5),
        tf.zeros([10]),
      ])
      for (let i = 0; i < this.model.layers.length - 1; i++)
        this.model.layers[i].trainable = false
    }

    this.model.compile({
      loss: "categoricalCrossentropy",
      optimizer: "adam",
      metrics: ["accuracy"],
    })

    this.component.setState({ modelLoaded: true })
  }

  initialize_data() {}
}
