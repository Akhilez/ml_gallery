import React from "react"
import { projects } from "../../globals/data"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import NumberPaintCanvas from "./paint_canvas"
import { IconButton, Text } from "@chakra-ui/react"
import { MdRefresh } from "react-icons/all"
import { mlgApi } from "../../api"

export class PositionalCNN extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      predClass: null,
      predPosition: null,
      confidences: null,
      dataLoaded: false,
    }

    this.project = projects.positional_cnn

    this.paintCanvasRef = React.createRef()
  }

  render() {
    return (
      <ProjectWrapper project={this.project} align="center">
        <NumberPaintCanvas
          ref={this.paintCanvasRef}
          parent={this}
          mt={6}
          mb={2}
        />
        <IconButton
          aria-label="icon"
          icon={<MdRefresh />}
          isRound
          variant="outline"
          colorScheme="red"
          size="sm"
          my={4}
          onClick={() => this.paintCanvasRef.current.clearCanvas()}
        />
        {this.state.predPosition && (
          <>
            <Text>
              <strong>Position: </strong>
              {this.state.predPosition}
            </Text>
            <Text>
              <strong>Class: </strong>
              {this.state.predClass}
            </Text>
          </>
        )}
      </ProjectWrapper>
    )
  }

  predict = image => {
    mlgApi.positionalCnn(image).then(result => {
      this.setState({ predClass: result.class, predPosition: result.position })
    })
  }
}
