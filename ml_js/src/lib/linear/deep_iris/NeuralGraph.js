import React from "react"
import loadable from "@loadable/component"
import petal_png from "./images/petal.png"
import sepal_png from "./images/sepal.png"

const Sketch = loadable(() => import("react-p5"))

export default class NeuralGraphIris extends React.Component {
  constructor(props) {
    super(props)

    this.state = props.state

    this.width = 800
    this.height = 500

    this.cx = this.width / 2
    this.cy = this.height / 2

    this.layerSpacing = 60
    this.neuronSpacing = 50
    this.radius = 10
    this.windowPadding = 10

    this.flowerSide = 100
    this.classificationWidth = 100

    this.petalImg = null

    this.neuronUpdateClickActions = []
  }

  render() {
    return (
      <Sketch
        setup={(p5, parent) => this.setup(p5, parent)}
        draw={p5 => this.draw(p5)}
        preload={p5 => this.preload(p5)}
        mouseClicked={p5 => this.mouseClicked(p5)}
      />
    )
  }

  setup(p5, parent) {
    this.p5 = p5

    p5.createCanvas(this.width, this.height).parent(parent)
    p5.frameRate(10)

    p5.angleMode(p5.DEGREES)
    p5.noStroke()
    p5.fill("red")
  }

  draw(p5) {
    p5.background(255)

    let x_start = this.flowerSide

    this.draw_layer(4, x_start, -1, true)
    for (let i = 0; i < this.state.nNeurons.length; i++) {
      this.draw_layer(
        this.state.nNeurons[i],
        x_start + this.layerSpacing * (i + 1),
        i
      )
    }
    x_start = x_start + this.layerSpacing * (this.state.nNeurons.length + 1)
    this.draw_layer(3, x_start, -1, true)

    if (x_start + this.classificationWidth !== this.width) {
      this.width = x_start + this.classificationWidth
      this.p5.resizeCanvas(this.width, this.height)
    }
  }

  preload(p5) {
    this.petalImg = p5.loadImage(petal_png)
    this.sepalImg = p5.loadImage(sepal_png)
  }

  draw_layer(nNeurons, x, index, io = false) {
    let layerHeight = (nNeurons - 1) * this.neuronSpacing

    let yStart = this.cy - layerHeight / 2

    for (let i = 0; i < nNeurons; i++) {
      this.p5.ellipse(x, yStart + i * this.neuronSpacing, this.radius)
    }

    if (!io)
      this.drawUpdateNeuronsButtons(
        index,
        x,
        yStart + nNeurons * this.neuronSpacing
      )
  }

  drawUpdateNeuronsButtons(index, x, y) {
    const side = 20
    x -= side / 2
    y -= side
    this.p5.rect(x, y, side, side)
    this.p5.rect(x, y + side, side, side)

    this.p5.push()
    this.p5.textSize(20)
    this.p5.fill("white")
    this.p5.text("+", x + 4, y + 16)
    this.p5.text("-", x + 7, y + 16 + side)
    this.p5.pop()

    if (this.neuronUpdateClickActions.length < 2 * this.state.nNeurons.length) {
      this.neuronUpdateClickActions.push({
        x: x,
        y: y,
        h: side,
        w: side,
        action: () => this.props.actions.updateNeurons(index, 1),
      })
      this.neuronUpdateClickActions.push({
        x: x,
        y: y + side,
        h: side,
        w: side,
        action: () => {
          if (this.state.nNeurons[index] > 1)
            this.props.actions.updateNeurons(index, -1)
        },
      })
    }
  }

  drawFlower2({ petalWidth, petalHeight, sepalWidth, sepalHeight }) {
    this.p5.rect(
      0,
      this.cy - this.flowerSide / 2,
      this.flowerSide,
      this.flowerSide
    )
    this.drawSepal(0, this.cy - this.flowerSide / 2, sepalHeight, sepalWidth)
    this.drawPetal(
      this.flowerSide / 2,
      this.cy - this.flowerSide / 2,
      petalHeight,
      petalWidth
    )
    this.drawSepal(
      this.flowerSide,
      this.cy - this.flowerSide / 2,
      sepalHeight,
      sepalWidth
    )

    this.drawPetal(0, this.cy, petalHeight, petalWidth)
    this.drawFlowerCenter(this.flowerSide / 2, this.cy, sepalHeight, sepalWidth)
    this.drawPetal(this.flowerSide, this.cy, petalHeight, petalWidth)

    this.drawSepal(0, this.cy + this.flowerSide / 2, sepalHeight, sepalWidth)
    this.drawPetal(
      this.flowerSide / 2,
      this.cy + this.flowerSide / 2,
      petalHeight,
      petalWidth
    )
    this.drawSepal(
      this.flowerSide,
      this.cy + this.flowerSide / 2,
      sepalHeight,
      sepalWidth
    )
  }

  drawSepal(x, y, height, width) {
    const color = this.p5.color("green")
    color.setAlpha(100)
    this.p5.fill(color)
    this.p5.circle(x, y, height)
  }

  drawPetal(x, y, height, width) {
    const color = this.p5.color("red")
    color.setAlpha(100)
    this.p5.fill(color)
    this.p5.circle(x, y, height)
  }

  drawFlowerCenter(x, y, height, width) {
    const color = this.p5.color("black")
    color.setAlpha(100)
    this.p5.fill(color)
    this.p5.circle(x, y, height)
  }

  drawFlower() {
    this.p5.rect(
      0,
      this.cy - this.flowerSide / 2,
      this.flowerSide,
      this.flowerSide
    )

    let cx = this.flowerSide / 2

    this.p5.image(
      this.petalImg,
      cx - this.state.petalWidth / 2,
      this.cy - this.state.petalHeight,
      this.state.petalWidth,
      this.state.petalHeight
    )

    this.p5.image(
      this.sepalImg,
      cx - this.state.sepalWidth / 2,
      this.cy,
      this.state.sepalWidth,
      this.state.sepalHeight
    )
  }

  drawClassificationBox(x) {
    this.p5.rect(
      x,
      this.cy - this.classificationWidth / 2,
      this.classificationWidth,
      this.classificationWidth
    )
  }

  mouseClicked(p5) {
    this.neuronUpdateClickActions.forEach(item => {
      if (
        p5.mouseX > item.x &&
        p5.mouseX < item.x + item.w &&
        p5.mouseY > item.y &&
        p5.mouseY < item.y + item.h
      ) {
        item.action()
        this.neuronUpdateClickActions = []
      }
    })
  }
}
