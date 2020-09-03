import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"

export class MnistGan extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.mnist_gan
  }
  render() {
    return <ProjectWrapper project={this.project} />
  }
}
