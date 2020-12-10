import React from "react"
import { projects } from "../../globals/data"
import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import NumberPaintCanvas from "./paint_canvas"
import { mlgApi } from "../../api"

export class PositionalCNN extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      predClass: null,
      predPosition: null,
      confidences: null,
      dataLoaded: false,
    }

    this.project = projects.positional_cnn

    this.paintCanvasRef = React.createRef()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <NumberPaintCanvas ref={this.paintCanvasRef} parent={this} mt={6} />
          {this.state.predClass}
          {this.state.predPosition}
        </Centered>
      </ProjectWrapper>
    )
  }

  predict = image => {
    console.log(image)
    mlgApi
      .positionalCnn(image)
      .then(response => response.json())
      .then(result => {
        this.setState({
          predClass: result.class,
          predPosition: result.position,
        })
      })
  }
}
