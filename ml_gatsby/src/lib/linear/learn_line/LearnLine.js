import React from "react"
import { Text, Box, Flex } from "@chakra-ui/core"
import { projects } from "src/lib/globals/data"
import { ProjectWrapper } from "../../globals/GlobalWrapper"

export default class LearnLine extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.learn_line
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Text>{this.project.title}</Text>
      </ProjectWrapper>
    )
  }
}
