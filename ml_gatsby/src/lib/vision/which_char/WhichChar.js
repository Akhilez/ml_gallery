import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import MnistClassifier from "./classifier"
import { Centered } from "../../components/commons"
import { MdRefresh } from "react-icons/all"
import NumberPaintCanvas from "./paint_canvas"
import { Box } from "@chakra-ui/core"
import { Bar, BarChart, Tooltip, XAxis, YAxis } from "recharts"

export class WhichChar extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      isTraining: false,
      lossData: [],
      modelLoaded: false,
      predicted: null,
      confidences: [],
    }

    this.project = projects.which_char
    this.paintCanvasRef = React.createRef()
    this.convNet = new MnistClassifier(this)
  }

  componentDidMount() {
    this.convNet.initialize_model()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          {!this.state.modelLoaded && "Loading model..."}

          {this.state.modelLoaded && (
            <>
              <NumberPaintCanvas ref={this.paintCanvasRef} parent={this} />
              <MdRefresh
                onClick={() => this.paintCanvasRef.current.clearCanvas()}
              />
              Predicted: {this.state.predicted}
              <this.PredictionChart />
            </>
          )}
        </Centered>
      </ProjectWrapper>
    )
  }

  startTraining() {}

  stopTraining() {}

  PredictionChart = () => {
    const confidences = this.state.confidences?.map((confidence, index) => {
      return { confidence, label: index }
    })

    return (
      <Box>
        <BarChart width={200} height={100} data={confidences}>
          <XAxis type="category" hide />
          <YAxis type="number" hide />
          <Tooltip />
          <Bar label dataKey="confidence" nameKey="label" fill="#f62252" />
        </BarChart>
      </Box>
    )
  }
}
