import React from "react"
import { ProjectWrapper } from "src/lib/components/ProjectWrapper"
import { projects } from "../../globals/data"
import { Centered } from "../../components/commons"
import Neuron from "../learn_line/neuron"
import { Graph } from "./Graph"
import { Button } from "@chakra-ui/core"

export class LinearClassifier extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = projects.linear_classifier
    this.state = { isTraining: false }
    this.graphRef = React.createRef()
    this.neuronRef = React.createRef()
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <Neuron ref={this.neuronRef} />
          <Button
            variantColor="brand"
            borderRadius="lg"
            m={1}
            onClick={() => {
              this.graphRef.current.startTraining()
              this.setState({ isTraining: true })
            }}
          >
            TRAIN
          </Button>
          {this.state.isTraining && (
            <Button
              m={1}
              variant="outline"
              borderRadius="lg"
              variantColor="brand"
              onClick={() => {
                this.graphRef.current.stopTraining()
                this.setState({ isTraining: false })
              }}
            >
              STOP
            </Button>
          )}
          <Button
            m={1}
            variant="outline"
            variantColor="brand"
            borderRadius="lg"
            onClick={() => this.graphRef.current.removeData()}
          >
            CLEAR DATA
          </Button>
          <Graph ref={this.graphRef} neuronRef={this.neuronRef} />
        </Centered>
      </ProjectWrapper>
    )
  }
}
