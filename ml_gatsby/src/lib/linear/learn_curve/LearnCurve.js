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
  ResponsiveContainer,
} from "recharts"
import { projects } from "../../globals/data"
import {
  Button,
  Flex,
  Alert,
  AlertIcon,
  CloseButton,
  Box,
} from "@chakra-ui/react"
import { LearnCurveTF } from "./LearnCurveTF"

export class LearnCurve extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      setState: state => this.setState(state),
      order: 5,
      isTraining: false,
      warningMessage: null,
      lossData: [],
    }

    this.project = projects.learn_curve

    this.tf = new LearnCurveTF()
    this.tf.setOrder(this.state.order)

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
              colorScheme="brand"
              borderRadius="lg"
              m={1}
              isLoading={this.state.isTraining}
              loadingText="Training"
              onClick={() => this.startTraining()}
            >
              TRAIN
            </Button>
          )}
          {this.state.isTraining && (
            <Button
              m={1}
              variant="outline"
              colorScheme="brand"
              borderRadius="lg"
              onClick={() => this.stopTraining()}
            >
              STOP
            </Button>
          )}
          {!this.state.isTraining && (
            <Button
              m={1}
              variant="outline"
              colorScheme="brand"
              borderRadius="lg"
              onClick={() => this.clearData()}
            >
              CLEAR
            </Button>
          )}

          {this.state.warningMessage && (
            <Alert status="warning" mt={4} w="md">
              <AlertIcon />
              {this.state.warningMessage}
              <CloseButton
                position="absolute"
                right="8px"
                top="8px"
                onClick={() => this.setState({ warningMessage: null })}
              />
            </Alert>
          )}

          {this.getComplexityModifier()}

          <br />
          <Graph
            ref={this.graphRef}
            tf={this.tf}
            new_point_classback={(x, y) => this.tf.addNewPoint(x, y)}
          />
          {this.getLossGraph()}
        </Centered>
      </ProjectWrapper>
    )
  }

  getComplexityModifier() {
    let terms = [<Box whiteSpace="nowrap">y =&nbsp;</Box>]

    for (let i = 1; i <= this.state.order; i++) {
      terms.push(
        <Box key={`eqn-${i}`} whiteSpace="nowrap">
          w<sub>{i}</sub>x<sup>{this.state.order - i + 1}</sup> + &nbsp;
        </Box>
      )
    }
    terms.push(<>b</>)

    return (
      <Box fontSize="20" mt={10}>
        <Box
          display="flex"
          fontSize={28}
          mb={5}
          overflow="auto"
          justifyContent={{ md: "center" }}
        >
          {terms.map(item => item)} <br />
        </Box>
        Change Complexity:
        <Button
          variant="outline"
          colorScheme="brand"
          ml={2}
          size="sm"
          disabled={this.state.isTraining}
          onClick={() => this.changeOrder(1)}
        >
          +
        </Button>
        {this.state.order > 1 && (
          <Button
            variant="outline"
            colorScheme="brand"
            ml={2}
            size="sm"
            disabled={this.state.isTraining}
            onClick={() => this.changeOrder(-1)}
          >
            -
          </Button>
        )}
      </Box>
    )
  }

  startTraining() {
    if (this.tf.data == null)
      return this.setState({
        warningMessage: "Click the box below to add training data",
      })
    this.setState({ isTraining: true })
    this.tf.train(10000, logs =>
      this.setState({
        lossData: this.state.lossData.concat([
          { index: this.state.lossData.length, loss: logs.loss },
        ]),
      })
    )
  }

  stopTraining() {
    this.setState({ isTraining: false })
    this.tf.stopTraining()
  }

  changeOrder(change) {
    // change => integer that is the diff of current and the new
    this.setState({ order: this.state.order + change }, () => {
      this.tf.setOrder(this.state.order)
    })
  }

  clearData() {
    this.graphRef.current.x = []
    this.graphRef.current.y = []

    this.tf.data = null
  }

  getLossGraph() {
    return (
      <Box maxW="500px">
        <ResponsiveContainer width="100%" height={300}>
          <LineChart
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
        </ResponsiveContainer>
      </Box>
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
