import React from "react"
import { isCursorInScope } from "../../utils/utils"
import loadable from "@loadable/component"
const Sketch = loadable(() => import("react-p5"))

export default class Graph extends React.Component {
  constructor(props) {
    super(props)

    this.height = 800
    this.width = 800

    this.pointDiameter = 10

    this.x = []
    this.y = []

    this.weights = [0, 0]

    this.cx = this.height / 2
    this.cy = this.width / 2

    this.p5 = null
  }

  render() {
    return (
      <div className={"rounded"} style={{ marginTop: 30 }}>
        <Sketch
          setup={(p5, parent) => this.setup(p5, parent)}
          draw={p5 => this.draw(p5)}
          mouseClicked={p5 => this.handleInput(p5)}
        />
      </div>
    )
  }

  setup(p5, parent) {
    this.p5 = p5

    p5.createCanvas(this.width, this.height).parent(parent)
    p5.frameRate(10)
  }

  draw(p5) {
    p5.background(243)
    this.drawAxes()

    for (let i = 0; i < this.x.length; i++) {
      this.drawPoint(this.x[i], this.y[i])
    }

    this.drawCurve()
  }

  handleInput(p5) {
    if (!isCursorInScope(p5, this.height, this.width)) return

    this.x.push(p5.mouseX)
    this.y.push(p5.mouseY)
    let [x, y] = this.lengthsToCoordinates(p5.mouseX, p5.mouseY)
    this.props.new_point_classback(x, y)
  }

  drawCurve() {
    let x = []
    let increment = 0.01
    for (let i = -1; i <= 1; i += increment) x.push(i)
    let y = this.getFunctionOutput(x)

    this.p5.stroke("red")
    this.p5.noFill()
    this.p5.strokeWeight(1)
    this.p5.beginShape()
    for (let i = 0; i < x.length; i++) {
      let [x_loc, y_loc] = this.coordinatesToLengths(x[i], y[i])
      this.p5.vertex(x_loc, y_loc)
      // this.drawPoint(x_loc, y_loc);
      // console.log(x_loc, y_loc);
    }
    this.p5.endShape()

    // exit();
  }

  drawAxes() {
    this.p5.stroke(220)
    this.p5.strokeWeight(1)
    this.p5.line(0, this.cy, this.width, this.cy)
    this.p5.line(this.cx, 0, this.cx, this.height)
  }

  getFunctionOutput(x) {
    let y = []
    let n = this.weights.length - 1
    for (let xi of x) {
      let yi = this.weights[n]
      for (let i = 0; i < n; i++) {
        yi += Math.pow(xi, n - i) * this.weights[i]
      }
      y.push(yi)
    }
    return y
  }

  drawPoint(x, y) {
    this.p5.fill("red")
    this.p5.noStroke()
    this.p5.ellipse(x, y, this.pointDiameter, this.pointDiameter)
  }

  lengthsToCoordinates(x, y) {
    let new_x = (x - this.cx) / this.cx
    let new_y = (y - this.cy) / this.cy

    // console.log(`LocX: ${x}, LocY: ${y}\ncoordX: ${new_x}, coordY: ${new_y}`);

    return [new_x, new_y]
  }

  coordinatesToLengths(x, y) {
    let new_x = this.cx + this.cx * x
    let new_y = this.cy + this.cy * y

    // console.log(`coordX: ${x}, coordY: ${y}\nLocX: ${new_x}, LocY: ${new_y}`);

    return [new_x, new_y]
  }
}
