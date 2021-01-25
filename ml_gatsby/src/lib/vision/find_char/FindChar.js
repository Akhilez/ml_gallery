import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { Centered } from "../../components/commons"
import { Flex, FormLabel, Switch, IconButton } from "@chakra-ui/react"
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
    this.autoClearEnabled = true

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
              <Flex justify="center" align="center" mt={4}>
                <FormLabel htmlFor="autoClear">Auto clear</FormLabel>
                <Switch
                  id="autoClear"
                  color="red"
                  defaultIsChecked
                  onClick={event => {
                    if (event.target.checked != null)
                      this.autoClearEnabled = event.target.checked
                  }}
                />
              </Flex>
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
            </>
          )}
        </Centered>
      </ProjectWrapper>
    )
  }
}
