import React from "react"
import { projects } from "../../globals/data"
import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"

export class PositionalCNN extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      predicted: null,
      confidences: null,
      dataLoaded: false,
    }

    this.project = projects.positional_cnn

    this.paintCanvasRef = React.createRef()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>Coming soon!</Centered>
      </ProjectWrapper>
    )
  }
}
