import React from "react"
import loadable from "@loadable/component"
import { isCursorInScope } from "src/lib/utils/utils"
import {
  Box,
  Popover,
  PopoverTrigger,
  PopoverContent,
  PopoverArrow,
  PopoverBody,
  PopoverCloseButton,
} from "@chakra-ui/core"

const Sketch = loadable(() => import("react-p5"))
// import Sketch from "react-p5";

export default class NumberPaintCanvas extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      isPopupOpen: true,
    }
    this.side = 140
    this.clearPaint = false
  }

  render() {
    return (
      <Popover
        isOpen={this.state.isPopupOpen}
        onClose={() => this.setState({ isPopupOpen: false })}
        placement="top"
        initialFocusRef={this}
      >
        <PopoverTrigger>
          <Box {...this.props} w="144px" border="2px solid red">
            <Sketch
              setup={(p5, parent) => this.setup(p5, parent)}
              draw={p5 => this.draw(p5)}
              mouseReleased={p5 => this.mouseReleased(p5)}
            />
          </Box>
        </PopoverTrigger>
        <PopoverContent zIndex={40}>
          <PopoverArrow />
          <PopoverCloseButton />
          <PopoverBody>Draw a number in this box</PopoverBody>
        </PopoverContent>
      </Popover>
    )
  }

  setup(p5, parent) {
    this.p5 = p5
    p5.createCanvas(this.side, this.side).parent(parent)
    p5.frameRate(60)
    p5.colorMode(p5.RGB, 255)
    p5.pixelDensity(1)
    p5.background(255, 255, 255)
    // p5.filter(p5.BLUR, 2)
  }

  draw(p5) {
    if (p5.mouseIsPressed) {
      if (p5.mouseButton === p5.LEFT) {
        p5.strokeWeight(15)
        p5.stroke(0)
        p5.line(p5.mouseX, p5.mouseY, p5.pmouseX, p5.pmouseY)
      }
    }
  }

  clearCanvas() {
    this.p5.background(255, 255, 255)
  }

  mouseReleased(p5) {
    if (!isCursorInScope(p5, this.side, this.side)) return

    // p5.filter(p5.BLUR, 4)

    p5.loadPixels()
    this.props.parent.convNet.predict(p5.pixels)
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
