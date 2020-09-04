import React from "react"
import loadable from "@loadable/component"

const Sketch = loadable(() => import("react-p5"))

export default class Neuron extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      w: null,
      b: null,
    }

    this.height = 300
    this.width = 600

    this.cx = this.width / 2
    this.cy = this.height / 2

    this.lineLength = 150
  }

  render() {
    return (
      <Sketch
        setup={(p5, parent) => this.setup(p5, parent)}
        draw={p5 => this.draw(p5)}
      />
    )
  }

  set(state) {
    this.setState(state)
  }

  setup(p5, parent) {
    p5.createCanvas(this.width, this.height).parent(parent)
    p5.frameRate(10)
  }

  draw(p5) {
    p5.background(243)
    p5.textSize(18)

    // Weight
    if (this.state.w < 0) p5.stroke(247, 120, 35)
    else p5.stroke(235, 16, 93)
    if (this.state.w != null) {
      p5.strokeWeight(1)
      p5.text(`m = ${this.state.w.toFixed(3)}`, this.cx - 110, this.cy - 50)
      p5.strokeWeight(this.rescale(this.state.w))
    } else {
      p5.strokeWeight(1)
      p5.text(`m`, this.cx - 110, this.cy - 50)
    }
    p5.line(this.cx, this.cy, this.cx - this.lineLength, this.cy - 50)

    // Bias
    if (this.state.b < 0) p5.stroke(247, 120, 35)
    else p5.stroke(235, 16, 93)
    if (this.state.b != null) {
      p5.strokeWeight(1)
      p5.text(`c = ${this.state.b.toFixed(3)}`, this.cx - 110, this.cy + 60)
      p5.strokeWeight(this.rescale(this.state.b))
    } else {
      p5.strokeWeight(1)
      p5.text(`c`, this.cx - 110, this.cy + 60)
    }
    p5.line(this.cx, this.cy, this.cx - this.lineLength, this.cy + 50)

    // y
    p5.stroke(100, 100, 100)
    p5.strokeWeight(1)
    p5.line(this.cx, this.cy, this.cx + this.lineLength, this.cy)

    // Circle
    p5.fill(235, 16, 93)
    p5.noStroke()
    p5.ellipse(this.cx, this.cy, 50)

    // x
    p5.text("x", this.cx - this.lineLength - 20, this.cy - 45)

    // 1
    p5.text("1", this.cx - this.lineLength - 20, this.cy + 55)

    // y
    p5.text("y", this.cx + this.lineLength + 20, this.cy + 5)
  }

  rescale(t) {
    return 5 * Math.tanh(t) ** 2 + 0.1
  }
}
