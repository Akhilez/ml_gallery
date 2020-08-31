import React from "react"
import { ProjectWrapper } from "src/lib/globals/GlobalWrapper"
import { projects } from "../../globals/data"
import { Text } from "@chakra-ui/core"

export class LinearClassifier extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = projects.linear_classifier
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Text>Classify these points</Text>
      </ProjectWrapper>
    )
  }
}
