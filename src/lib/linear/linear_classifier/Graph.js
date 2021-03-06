import LinearClassifierNeuron from "./Neuron"
import TrainingTracker from "../../utils/training_tracker"
import loadable from "@loadable/component"
import React from "react"
import Chartist from "../../utils/chartist"
import { isCursorInScope } from "src/lib/utils/utils"
import { Box } from "@chakra-ui/react"

const Sketch = loadable(() => import("react-p5"))

export class Graph extends React.Component {
  constructor(props) {
    super(props)
    this.neuron = new LinearClassifierNeuron()
    this.state = {
      isTraining: false,
    }

    this.height = 500
    this.width = 800

    this.tracker = new TrainingTracker()
    this.chartist = null

    this.neuronRef = props.neuronRef
  }

  render() {
    return (
      <Box mt={10} overflow="auto">
        <Sketch
          setup={(p5, parent) => this.setup(p5, parent)}
          draw={p5 => this.draw(p5)}
          mouseClicked={p5 => this.handleInput(p5)}
        />
      </Box>
    )
  }

  setup(p5, parent) {
    p5.createCanvas(this.width, this.height).parent(parent)
    p5.frameRate(this.tracker.frameRate)
    this.chartist = new Chartist(p5, this.width, this.height)
  }

  draw(p5) {
    p5.clear()

    this.chartist.drawPoints(this.neuron.getDataPoints())
    let params = this.neuron.getMC()
    this.chartist.drawLine(params.w, params.b)

    if (this.tracker.isComplete() || !this.state.isTraining) return

    this.tracker.updateFrame()

    if (this.tracker.isNewEpoch()) {
      this.neuron.fullPass()
      this.neuronRef.current.set({ w: params.w, b: params.b })
    }
  }

  handleInput(p5) {
    if (!isCursorInScope(p5, this.height, this.width)) return

    let label = 1
    if (p5.keyIsDown(p5.SHIFT)) label = 0
    let x = p5.mouseX / this.width
    let y = p5.mouseY / this.height
    this.neuron.addDataPoint(x, y, label)
  }

  startTraining() {
    this.setState({ isTraining: true })
    this.tracker.epoch = 0
  }

  stopTraining() {
    this.setState({ isTraining: false })
  }

  removeData() {
    this.neuron.removeData()
  }
}
