import React from "react"
import Sketch from "react-p5"
import { isCursorInScope } from "../../utils/utils"

export default class NumberPaintCanvas extends React.Component {
  constructor(props) {
    super(props)

    this.matrixSide = 28
    this.scale = 5
    this.side = this.matrixSide * this.scale
    this.radius = 2

    this.paintMatrix = this.getEmptyMatrix(this.matrixSide, this.matrixSide)

    this.clearPaint = false
  }

  render() {
    return (
      <div>
        <Sketch
          setup={(p5, parent) => this.setup(p5, parent)}
          draw={p5 => this.draw(p5)}
          mouseReleased={p5 => this.mouseReleased(p5)}
        />
      </div>
    )
  }

  setup(p5, parent) {
    this.p5 = p5

    p5.createCanvas(this.side, this.side).parent(parent)
    p5.frameRate(60)

    p5.background(255)
  }

  draw(p5) {
    if (p5.mouseIsPressed) {
      if (p5.mouseButton === p5.LEFT) {
        p5.strokeWeight(15)
        // p5.stroke(255,0,0);
        p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY)
      }
    }
  }

  clearCanvas() {
    this.p5.background(255)
  }

  mouseReleased(p5) {
    console.log("Hey. The position is", p5.mouseX, p5.mouseY)
    if (!isCursorInScope(p5, this.side, this.side)) return

    p5.loadPixels()
    this.props.parent.convNet.captureP5Image(p5.pixels)
  }

  getEmptyMatrix(r, c) {
    let matrix = []
    for (let i = 0; i < r; i++) {
      let matrix_i = []
      for (let j = 0; j < c; j++) matrix_i.push(0)
      matrix.push(matrix_i)
    }
    return matrix
  }
}
