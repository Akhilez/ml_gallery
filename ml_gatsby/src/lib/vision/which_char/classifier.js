import * as tf from "@tensorflow/tfjs"
import * as tfvis from "@tensorflow/tfjs-vis"

const modelUrl =
  "https://storage.googleapis.com/akhilez/models/mnist_classifier/model.json"
const dataUrl =
  "https://storage.googleapis.com/akhilez/datasets/mnist/data_1000.json"

export default class MnistClassifier {
  constructor(component) {
    this.component = component
    this.model = null
    this.data = null
  }

  async initialize_model() {
    this.model = await tf.loadLayersModel(modelUrl)

    const lastLayer = this.model.layers[this.model.layers.length - 1]
    lastLayer.setWeights([
      tf.randomUniform([100, 10], 0.5, -0.5),
      tf.zeros([10]),
    ])
    for (let i = 0; i < this.model.layers.length - 1; i++)
      this.model.layers[i].trainable = false

    this.model.compile({
      loss: "categoricalCrossentropy",
      optimizer: "adam",
      metrics: ["accuracy"],
    })

    this.component.setState({ modelLoaded: true })
  }

  async initialize_data() {
    const response = await fetch(dataUrl)
    const jsonData = await response.json()

    const images = tf.tensor(jsonData.images).reshape([-1, 28, 28, 1])
    const classes = tf.tensor(jsonData.classes)

    this.data = [images, classes]
    this.component.setState({ dataLoaded: true }, () => this.drawSamples())
  }

  async train() {
    this.model.fit(this.data[0], this.data[1], {
      epochs: 1000,
      batchSize: 4,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          this.model.stopTraining = !this.component.state.isTraining
          this.model.layers[0].getWeights()[1].print()
        },
      },
    })
  }

  pixToTensor(pixels) {
    return tf
      .scalar(255)
      .sub(tf.tensor(Array.from(pixels)))
      .div(255)
      .reshape([140, 140, 4])
      .split(4, 2)[0]
      .resizeBilinear([28, 28])
      .reshape([1, 28, 28, 1])
  }

  predict(pixels) {
    let image = this.pixToTensor(pixels)

    const output = this.model.predict(image)
    const confidences = output.mul(100).squeeze().arraySync()
    const predicted = output.argMax(1).dataSync()

    this.component.setState({ predicted: predicted, confidences: confidences })
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

  drawSamples() {
    const images = tf
      .scalar(1.0)
      .sub(this.data[0])
      .reshape([-1, 28, 28])
      .arraySync()
    const labels = this.data[1].arraySync()
    const data = {}
    for (let i = 0; i < 1000; i++) {
      const index = Math.floor(Math.random() * images.length)
      const label = tf.tensor(labels[index]).argMax().dataSync()[0]
      if (data[label] != null) continue
      data[label] = images[index]
      if (Object.keys(data).length === 10) {
        for (let j = 0; j < 10; j++) {
          let img = tf.tensor(data[j])
          img = img.reshape([28, 28, 1])
          img = img.resizeBilinear([
            this.component.sampleSide,
            this.component.sampleSide,
          ])
          tf.browser.toPixels(img, this.component.sampleRefs[j].current)
          this.component.sampleData[j] = img
            .mul(255)
            .resizeBilinear([140, 140])
            .reshape([140, 140])
            .transpose()
            .arraySync()
        }
        return
      }
    }
  }
}
