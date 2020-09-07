import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"

export default class MnistClassifier {
  constructor(component) {
    this.component = component
    this.model = null
  }

  async initialize_model() {
    this.model = await tf.loadLayersModel(
      "https://storage.googleapis.com/akhilez/models/mnist_classifier/model.json"
    )
    const lastLayer = this.model.layers[this.model.layers.length - 1]
    console.log(lastLayer.getWeights())
    lastLayer.setWeights([
      tf.randomUniform([100, 10], 0.5, -0.5),
      tf.zeros([10]),
    ])
    console.log(lastLayer)
    this.component.setState({ modelLoaded: true })
  }

  captureP5Image(pixels) {
    let image = tf // TODO: Figure out why RGBA is not working in linux.
      .scalar(255)
      .sub(tf.tensor(Array.from(pixels)))
      .div(255)
      .reshape([280, 280, 4])
      .split(4, 2)[0]
      .resizeBilinear([28, 28])
      .reshape([1, 28, 28, 1])

    let output = this.model.predict(image)
    let predicted = output.argMax(1).dataSync()
    output.print()
    this.component.setState({ predicted: predicted })
  }

  async plotImage(image) {
    let surface = tfvis
      .visor()
      .surface({ name: "Captured Image", tab: "capturedImage" })
    const canvas = document.createElement("canvas")
    canvas.width = 28
    canvas.height = 28
    canvas.style = "margin: 4px;"
    await tf.browser.toPixels(image, canvas)
    surface.drawArea.appendChild(canvas)
    image.dispose()
  }
}
