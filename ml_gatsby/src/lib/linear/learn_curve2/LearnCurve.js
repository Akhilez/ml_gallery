import React from "react"
import { ProjectWrapper } from "src/lib/components/ProjectWrapper"
import { Centered } from "../../components/commons"
import AjaxTransporter from "../../utils/transporter/ajax_transporter"
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

export class LearnCurve extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      order: 5,
      isTraining: false,
      loss: null,
      lossData: [],
      isTrainerInitialized: false,
    }
    this.project = projects.learn_curve

    this.transporter = new AjaxTransporter("learn_curve", data =>
      this.receive_data(data)
    )

    this.x = null
    this.y = null

    this.graphRef = React.createRef()
    this.neuronRef = React.createRef()
  }

  componentDidMount() {
    this.transporter.init()
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          {!this.state.isTrainerInitialized && (
            <p>Connecting to webserver ...</p>
          )}

          <NeuronGraphLearnCurve ref={this.neuronRef} />

          {this.state.isTrainerInitialized && (
            <Button
              variantColor="brand"
              borderRadius="lg"
              m={1}
              onClick={() => this.startTraining()}
            >
              TRAIN
            </Button>
          )}
          {this.state.isTrainerInitialized && (
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
          {this.state.isTrainerInitialized && (
            <Button
              m={1}
              variant="outline"
              variantColor="brand"
              borderRadius="lg"
              onClick={() => this.clearData()}
            >
              CLEAR
            </Button>
          )}

          {this.state.isTrainerInitialized && this.getComplexityModifier()}

          <br />
          <Graph
            ref={this.graphRef}
            new_point_classback={(x, y) => this.add_new_point(x, y)}
          />
          {this.state.isTrainerInitialized && this.getLossGraph()}
        </Centered>
      </ProjectWrapper>
    )
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
    let payload = {
      action: "start_training",
    }
    this.transporter.send(payload)
    this.setState({ isTraining: true })

    let count = 0
    let listener = setInterval(() => {
      if (count > 100) {
        this.transporter.send({ action: "stop_training" })
        this.setState({ isTraining: false })
        clearInterval(listener)
      } else if (this.state.isTraining) {
        this.transporter.send({ action: "listen" })
      }
      count++
    }, 1000)
  }

  stopTraining() {
    let payload = {
      action: "stop_training",
    }
    this.transporter.send(payload)
    this.setState({ isTraining: false })
  }

  receive_data(data) {
    if (data.action === "status_update") {
      this.updateTrainingStatus(data.data)
    } else if (data.action === "init") {
      this.transporter.job_id = data.job_id
      this.x = data.data[0]
      this.y = data.data[1]
      this.setState({ isTrainerInitialized: true })
      this.drawDataToCanvas(this.x, this.y)
    }
  }

  updateTrainingStatus(data) {
    this.setState({
      loss: data.train_error,
      lossData: this.state.lossData.concat([
        { index: this.state.lossData.length, loss: data.train_error },
      ]),
    })
    this.graphRef.current.weights = data.weights
    this.neuronRef.current.weights = data.weights

    if (data.is_training === false) this.setState({ isTraining: false })
  }

  changeOrder(change) {
    if (this.state.order <= 1 && change < 0) return

    let newOrder = this.state.order + change

    this.setState({ order: newOrder })

    this.neuronRef.current.initializeWeights(newOrder)

    this.transporter.send({
      action: "change_order",
      data: newOrder,
    })
  }

  clearData() {
    this.graphRef.current.x = []
    this.graphRef.current.y = []

    this.transporter.send({
      action: "clear_data",
    })
  }

  add_new_point(x, y) {
    this.transporter.send({
      action: "new_point",
      data: {
        x: x,
        y: y,
      },
    })
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
