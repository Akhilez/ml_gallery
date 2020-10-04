import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import IrisNet from "./IrisNet"
import NeuralGraph from "./NeuralGraph"
import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"
import { Centered } from "../../components/commons"
import { Box, Button } from "@chakra-ui/core"

export class DeepIris extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.deep_iris
    this.state = {
      nNeurons: [6, 4],
      isTraining: false,
      lossData: [],
    }

    this.graphRef = React.createRef()
    this.irisNet = new IrisNet(this)
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <NeuralGraph
            ref={this.graphRef}
            appState={this.state}
            actions={{
              updateNeurons: (layerNumber, change) =>
                this.updateNeurons(layerNumber, change),
            }}
          />

          {this.UpdateLayersButtons()}

          {!this.state.isTraining && (
            <Button
              variantColor="brand"
              borderRadius="lg"
              m={1}
              onClick={() => this.startTraining()}
            >
              TRAIN
            </Button>
          )}
          {this.state.isTraining && (
            <Button
              m={1}
              variant="outline"
              variantColor="brand"
              borderRadius="lg"
              onClick={() => this.stopTraining()}
            >
              STOP
            </Button>
          )}

          <br />

          {this.getLossGraph()}
        </Centered>
      </ProjectWrapper>
    )
  }

  getLossGraph() {
    return (
      <LineChart
        width={500}
        height={300}
        data={this.state.lossData}
        margin={{
          top: 5,
          right: 30,
          left: 20,
          bottom: 5,
        }}
      >
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="index" type="number" scale="auto" />
        <YAxis />
        <Tooltip />
        <Legend />
        <Line type="monotone" dataKey="loss" stroke="#8884d8" />
      </LineChart>
    )
  }

  startTraining() {
    this.setState({ isTraining: true })
    this.irisNet.train()
  }

  stopTraining() {
    this.setState({ isTraining: false })
  }

  updateNeurons(layerNumber, change) {
    this.state.nNeurons[layerNumber] += change
    this.setState({ nNeurons: this.state.nNeurons })
    this.irisNet.initialize_net()
  }

  updateLayers(change) {
    if (change > 0) {
      this.state.nNeurons.push(3)
      this.setState({ nNeurons: this.state.nNeurons })
    } else {
      this.state.nNeurons.pop()
      this.setState({ nNeurons: this.state.nNeurons })
    }
    this.irisNet.initialize_net()
  }

  UpdateLayersButtons() {
    return (
      <Box mb={4}>
        Change Depth:
        <Button
          variant="outline"
          variantColor="brand"
          ml={2}
          size="sm"
          disabled={() => this.state.isTraining}
          onClick={() => this.updateLayers(1)}
        >
          +
        </Button>
        <Button
          variant="outline"
          variantColor="brand"
          ml={2}
          size="sm"
          disabled={() => this.state.isTraining}
          onClick={() => this.updateLayers(-1)}
        >
          -
        </Button>
      </Box>
    )
  }
}
