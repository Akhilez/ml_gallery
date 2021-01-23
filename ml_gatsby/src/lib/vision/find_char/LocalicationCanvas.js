import React from "react"
import { Box } from "@chakra-ui/react"
import loadable from "@loadable/component"
import { isCursorInScope } from "../../utils/utils"

const Sketch = loadable(() => import("react-p5"))

export class LocalizationCanvas extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      isPopupOpen: true,
    }
    this.clearPaint = false
    this.p5 = null

    this.side = 112 * 4 // = 448
    this.anchor_cx = 56 * 4
    this.anchor_cy = 56 * 4
    this.anchor_w = 28 * 4

    this.isBeingDrawn = false
  }

  render() {
    return (
      <Box {...this.props} w="452px" border="2px solid red" mt={4}>
        <Sketch
          setup={(p5, parent) => this.setup(p5, parent)}
          draw={p5 => this.draw(p5)}
          mouseReleased={p5 => this.mouseReleased(p5)}
        />
      </Box>
    )
  }

  setup(p5, parent) {
    this.p5 = p5
    p5.createCanvas(this.side, this.side).parent(parent)
    p5.frameRate(60)
    p5.colorMode(p5.RGB, 255)
    p5.pixelDensity(1)
    p5.background(255, 255, 255)
  }

  draw(p5) {
    if (
      p5.mouseIsPressed &&
      p5.mouseButton === p5.LEFT &&
      isCursorInScope(this.p5, this.side, this.side)
    ) {
      if (this.props.parent.autoClearEnabled && !this.isBeingDrawn) {
        this.clearCanvas()
        this.isBeingDrawn = true
      }

      p5.strokeWeight(15)
      p5.stroke(0)
      p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY)
    }
  }

  drawBoundingBox(cxd, cyd, wd) {
    const cx = (0.5 + cxd) * this.side
    const cy = (0.5 + cyd) * this.side
    const w = (this.anchor_w / this.side + wd) * this.side

    const x = cx - w / 2
    const y = cy - w / 2

    this.p5.push()
    this.p5.strokeWeight(1)
    this.p5.stroke("red")
    this.p5.noFill()
    this.p5.rect(y, x, w)
    this.p5.pop()
  }

  clearCanvas() {
    this.p5.background(255, 255, 255)
  }

  mouseReleased(p5) {
    if (!isCursorInScope(p5, this.side, this.side)) return

    // p5.filter(p5.BLUR, 4)

    p5.loadPixels()
    this.props.parent.convNet.predict(p5.pixels)
    this.isBeingDrawn = false
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
