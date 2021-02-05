import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Link, Text } from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { GridWorldCanvas } from "./GridWorldCanvas"

export class GridWorld extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.grid_world
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <Text>Yo! Lets play grid world!</Text>
          <GridWorldCanvas />
        </Centered>
      </ProjectWrapper>
    )
  }
}
