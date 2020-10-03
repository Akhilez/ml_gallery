import React from "react"
import { ProjectWrapper } from "src/lib/components/ProjectWrapper"
import { Centered } from "../../components/commons"
import NeuronGraphLearnCurve from "./neuron_graph_learn_curve"
import Graph from "./sketch_learn_curve"
import {
  LineChart,
  CartesianGrid,
  XAxis,
  YAxis,
  Legend,
  Line,
  Tooltip,
} from "recharts"
import { projects } from "../../globals/data"
import { Button, Flex } from "@chakra-ui/core"
import { LearnCurveTF } from "./LearnCurveTF"

export class LearnCurve extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      setState: state => this.setState(state),
      order: 5,
      isTraining: false,
      loss: null,
      lossData: [],
    }

    this.stateOps = this.getStateOps()

    this.project = projects.learn_curve

    this.tf = new LearnCurveTF(this.state)

    this.x = null
    this.y = null

    this.graphRef = React.createRef()
    this.neuronRef = React.createRef()
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <NeuronGraphLearnCurve ref={this.neuronRef} tf={this.tf} />

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
          <Button
            m={1}
            variant="outline"
            variantColor="brand"
            borderRadius="lg"
            onClick={() => this.clearData()}
          >
            CLEAR
          </Button>

          {this.getComplexityModifier()}

          <br />
          <Graph
            ref={this.graphRef}
            new_point_classback={(x, y) => this.add_new_point(x, y)}
          />
          {this.getLossGraph()}
        </Centered>
      </ProjectWrapper>
    )
  }

  getStateOps() {
    return {
      setIsTraining: isTraining => this.setState({ isTraining: isTraining }),
    }
  }

  getComplexityModifier() {
    let terms = [<>y = </>]

    for (let i = 1; i <= this.state.order; i++) {
      terms.push(
        <div key={`eqn-${i}`}>
          w<sub>{i}</sub>x<sup>{this.state.order - i + 1}</sup> + &nbsp;
        </div>
      )
    }
    terms.push(<>b</>)

    return (
      <div style={{ fontSize: 20, marginTop: 50 }}>
        <Flex fontSize={28} mb={5} justifyContent="center">
          {terms.map(item => item)} <br />
        </Flex>
        Change Complexity:
        <Button
          variant="outline"
          variantColor="brand"
          ml={2}
          size="sm"
          onClick={() => this.changeOrder(1)}
        >
          +
        </Button>
        <Button
          variant="outline"
          variantColor="brand"
          ml={2}
          size="sm"
          onClick={() => this.changeOrder(-1)}
        >
          -
        </Button>
      </div>
    )
  }

  startTraining() {
    this.stateOps.setIsTraining(true)
    // TODO: start training
  }

  stopTraining() {
    this.stateOps.setIsTraining(false)
    // TODO: Stop training
  }

  changeOrder(change) {
    // TODO: Change order
  }

  clearData() {
    this.graphRef.current.x = []
    this.graphRef.current.y = []

    // TODO: Clear data
  }

  add_new_point(x, y) {
    // TODO: Add new point
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

  drawDataToCanvas(x, y) {
    let newX = []
    let newY = []
    for (let i = 0; i < x.length; i++) {
      let [xi, yi] = this.graphRef.current.coordinatesToLengths(x[i], y[i])
      newX.push(xi)
      newY.push(yi)
    }
    this.graphRef.current.x = newX
    this.graphRef.current.y = newY
  }
}
