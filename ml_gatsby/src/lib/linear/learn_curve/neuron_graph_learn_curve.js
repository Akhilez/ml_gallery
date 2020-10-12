import React from "react"
import loadable from "@loadable/component"
import { Box } from "@chakra-ui/core"

const Sketch = loadable(() => import("react-p5"))

export default class NeuronGraphLearnCurve extends React.Component {
  constructor(props) {
    super(props)

    this.tf = props.tf

    this.width = 600
    this.height = 400

    this.cx = this.width / 2
    this.cy = this.height / 2

    this.weightLength = 120
    this.inputSpacing = 60
  }

  render() {
    return (
      <Box overflow="auto">
        <Sketch
          setup={(p5, parent) => this.setup(p5, parent)}
          draw={p5 => this.draw(p5)}
        />
      </Box>
    )
  }

  setup(p5, parent) {
    this.p5 = p5

    p5.createCanvas(this.width, this.height).parent(parent)
    p5.frameRate(10)
  }

  draw(p5) {
    p5.background(243)

    this.drawOutput()
    this.drawInputs()
    this.drawNeuron()
  }

  drawInputs() {
    const weights = this.tf.getWeights()
    let n = weights.length

    let leftX = this.cx - this.inputSpacing
    let rightX = this.cx + this.inputSpacing

    let positions = []

    for (let i = 0; i < n; i++) {
      if (i === 0) {
        positions.push(this.cx)
      } else if (i % 2 === 0) {
        positions.push(leftX)
        leftX -= this.inputSpacing
      } else {
        positions.push(rightX)
        rightX += this.inputSpacing
      }
    }

    positions.sort()

    for (let i = 0; i < positions.length; i++) {
      this.drawInputLine(positions[i], i, weights[i], n)
    }
  }

  drawInputLine(x, i, weight, len) {
    let y = this.cy + this.weightLength

    if (weight < 0) this.p5.stroke(247, 120, 35)
    else this.p5.stroke(235, 16, 93)

    this.p5.strokeWeight(this.rescale(weight))
    this.p5.line(x, y, this.cx, this.cy)

    this.p5.strokeWeight(1)
    this.p5.push()
    this.p5.translate(x, y - 20)
    this.p5.rotate(this.p5.radians(-20))
    this.p5.textSize(16)
    this.p5.text(weight.toFixed(3), 0, 0)
    this.p5.pop()

    this.p5.textSize(18)
    if (i === len - 1) this.p5.text("1", x, y + 20)
    else this.p5.text(`x^${len - i - 1}`, x - 10, y + 20)
  }

  drawNeuron() {
    this.p5.fill(235, 16, 93)
    this.p5.noStroke()
    this.p5.ellipse(this.cx, this.cy, 50)
  }

  drawOutput() {
    this.p5.stroke(100, 100, 100)
    this.p5.strokeWeight(1)
    this.p5.line(this.cx, this.cy, this.cx, this.cy - this.weightLength + 20)

    this.p5.text("y", this.cx - 2, this.cy - this.weightLength)
  }

  rescale(t) {
    return 5 * Math.tanh(t) ** 2 + 0.1
  }
}
