import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { Centered } from "../../components/commons"
import { Button } from "@chakra-ui/core"

export class FindChar extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.find_char

    this.job_id = null
    this.actions = {}
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <this.TrainButton />
        </Centered>
      </ProjectWrapper>
    )
  }

  TrainButton = () => {
    return <Button>TRAIN</Button>
  }
}
