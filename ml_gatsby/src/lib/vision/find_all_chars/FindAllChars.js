import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { Centered } from "../../components/commons"
import { IconButton } from "@chakra-ui/core"
import { FindAllCharsCanvas } from "./FindAllCharsCanvas"
import { FindAllCharsTF } from "./FindAllCharsTF"
import { MdRefresh } from "react-icons/all"

export class FindAllChars extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.find_all_chars

    this.state = {
      isTraining: false,
      lossData: [],
      modelLoaded: false,
      predicted: null,
      dataLoaded: false,
    }

    this.canvasRef = React.createRef()

    this.convNet = new FindAllCharsTF(this)
    this.convNet.initialize_model()
    // this.convNet.initialize_data()
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
            variantColor="red"
            size="sm"
            mt={4}
            onClick={() => this.canvasRef.current.clearCanvas()}
          />
        </Centered>
      </ProjectWrapper>
    )
  }
}
