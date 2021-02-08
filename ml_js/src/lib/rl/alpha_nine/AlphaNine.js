import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Link, Text } from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { AlphaNineCanvas } from "./AlphaNineCanvas"

export class AlphaNine extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.alpha_nine
  }

  render() {
    return (
      <ProjectWrapper project={this.project} align="center">
        <Text>
          This is a 2 player game. AI player coming soon. Built using my own 9
          Men's Morris{" "}
          <Link to="https://github.com/Akhilez/gyms">environment</Link>
        </Text>
        <AlphaNineCanvas />
      </ProjectWrapper>
    )
  }
}
