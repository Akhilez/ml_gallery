import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { Centered } from "../../components/commons"
import { IconButton } from "@chakra-ui/react"
import { FindAllCharsCanvas } from "./FindAllCharsCanvas"
import { MdRefresh } from "react-icons/all"

export class FindAllChars extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.find_all_chars

    this.state = {
      predicted: null,
    }

    this.canvasRef = React.createRef()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <FindAllCharsCanvas ref={this.canvasRef} parent={this} />
          <IconButton
            aria-label="icon"
            icon={MdRefresh}
            isRound
            variant="outline"
            colorScheme="red"
            size="sm"
            mt={4}
            onClick={() => this.canvasRef.current.clearCanvas()}
          />
        </Centered>
      </ProjectWrapper>
    )
  }

  predict = () => {}
}
