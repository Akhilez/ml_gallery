import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import MnistClassifier from "./classifier"
import { Centered } from "../../components/commons"
import { MdRefresh } from "react-icons/all"
import NumberPaintCanvas from "./paint_canvas"

export class WhichChar extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      isTraining: false,
      lossData: [],
      modelLoaded: false,
      predicted: null,
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
          <h1>Which Character?</h1>
          <p>Predict which number is being drawn.</p>
          <br />
          {!this.state.modelLoaded && (
            <>
              Loading model...
              <br />
            </>
          )}

          {this.state.modelLoaded && (
            <>
              <NumberPaintCanvas ref={this.paintCanvasRef} parent={this} />
              <div>
                <MdRefresh
                  onClick={() => this.paintCanvasRef.current.clearCanvas()}
                />
                <br />
                Predicted: {this.state.predicted}
              </div>
            </>
          )}
        </Centered>
      </ProjectWrapper>
    )
  }

  startTraining() {}

  stopTraining() {}
}
