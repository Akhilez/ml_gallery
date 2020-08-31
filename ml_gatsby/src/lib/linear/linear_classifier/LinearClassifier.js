import React from "react"
import { ProjectWrapper } from "src/lib/globals/GlobalWrapper"
import { projects } from "../../globals/data"
import { Centered } from "../../components/commons"
import Neuron from "../learn_line/neuron"
import { Graph } from "./Graph"

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
          <button
            className={"ActionButton"}
            onClick={() => {
              this.graphRef.current.startTraining()
              this.setState({ isTraining: true })
            }}
          >
            TRAIN
          </button>
          {this.state.isTraining && (
            <button
              className={"ActionButton"}
              onClick={() => {
                this.graphRef.current.stopTraining()
                this.setState({ isTraining: false })
              }}
            >
              STOP
            </button>
          )}
          <button
            className={"ActionButton"}
            onClick={() => this.graphRef.current.removeData()}
          >
            CLEAR DATA
          </button>
          <Graph ref={this.graphRef} neuronRef={this.neuronRef} />
        </Centered>
      </ProjectWrapper>
    )
  }
}
