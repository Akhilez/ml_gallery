import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { Centered } from "../../components/commons"
import { Button, IconButton } from "@chakra-ui/core"
import { LocalizationCanvas } from "./LocalicationCanvas"
import { FindCharTF } from "./FindCharTF"
import { MdRefresh } from "react-icons/all"

export class FindChar extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.find_char

    this.state = {
      isTraining: false,
      lossData: [],
      modelLoaded: false,
      predicted: null,
      dataLoaded: false,
    }

    this.canvasRef = React.createRef()

    this.convNet = new FindCharTF(this)
    this.convNet.initialize_model()
    // this.convNet.initialize_data()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          {!this.state.modelLoaded && "Loading model..."}
          {this.state.modelLoaded && (
            <>
              <LocalizationCanvas ref={this.canvasRef} parent={this} />
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
            </>
          )}
        </Centered>
      </ProjectWrapper>
    )
  }
}
