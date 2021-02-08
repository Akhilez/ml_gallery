import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import MnistClassifier from "./classifier"
import { MdRefresh } from "react-icons/all"
import NumberPaintCanvas from "./paint_canvas"
import { Box, Button, Flex, IconButton, Text } from "@chakra-ui/react"
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts"

export class WhichChar extends React.Component {
  constructor(props) {
    super(props)

    this.state = {
      isTraining: false,
      lossData: [],
      modelLoaded: false,
      predicted: null,
      confidences: null,
      dataLoaded: false,
    }

    this.project = projects.which_char
    this.paintCanvasRef = React.createRef()
    this.convNet = new MnistClassifier(this)
    this.convNet.initialize_model()
    this.convNet.initialize_data()

    this.sampleRefs = []
    for (let i = 0; i < 10; i++) this.sampleRefs.push(React.createRef())
    this.sampleSide = 100
    this.sampleData = {}
  }

  render() {
    return (
      <ProjectWrapper project={this.project} align="center">
        {!this.state.modelLoaded && "Loading model..."}

        {this.state.modelLoaded && (
          <>
            <NumberPaintCanvas ref={this.paintCanvasRef} parent={this} mt={6} />
            <IconButton
              aria-label="icon"
              icon={<MdRefresh />}
              isRound
              variant="outline"
              colorScheme="red"
              size="sm"
              mt={4}
              onClick={() => this.paintCanvasRef.current.clearCanvas()}
            />
            {this.state.predicted && (
              <Text my={2}>Predicted: {this.state.predicted}</Text>
            )}
            {this.state.confidences && <this.PredictionChart />}
            <Flex justifyContent="center" mt={4}>
              <Button
                colorScheme="brand"
                borderRadius="lg"
                m={1}
                isLoading={this.state.isTraining || !this.state.dataLoaded}
                loadingText={
                  this.state.isTraining ? "Training" : "Loading Data"
                }
                onClick={() => this.startTraining()}
              >
                TRAIN
              </Button>
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
            </Flex>
          </>
        )}
        <this.Samples />
        <this.LossGraph />
      </ProjectWrapper>
    )
  }

  Samples = () => {
    return (
      <Flex justifyContent={{ lg: "center" }} overflow="auto" py={4}>
        {this.sampleRefs.map((ref, index) => (
          <Box
            as="canvas"
            onClick={() => this.setSampleImage(index)}
            key={index}
            ref={ref}
            height={`${this.sampleSide}px`}
            width={`${this.sampleSide}px`}
            _hover={{ border: "5px solid red", cursor: "pointer" }}
            shadow="lg"
            mx={1}
            borderRadius={8}
          />
        ))}
      </Flex>
    )
  }

  LossGraph = () => {
    return (
      <Box maxW="500px" m={10}>
        <ResponsiveContainer width="100%" height={300}>
          <LineChart data={this.state.lossData}>
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

  startTraining() {
    this.setState({ isTraining: true })
    this.convNet.train()
  }

  stopTraining() {
    this.setState({ isTraining: false })
  }

  predict() {
    const p5 = this.paintCanvasRef.current.p5
    p5.loadPixels()
    this.convNet.predict(p5.pixels)
  }

  setSampleImage(index) {
    const imageData = this.sampleData[index]
    for (let i = 0; i < imageData.length; i++)
      for (let j = 0; j < imageData[i].length; j++)
        this.paintCanvasRef.current.p5.set(i, j, imageData[i][j])
    this.paintCanvasRef.current.p5.updatePixels()
    this.predict()
  }

  PredictionChart = () => {
    const confidences = this.state.confidences?.map((confidence, index) => {
      return { confidence, label: index }
    })

    return (
      <Box>
        <BarChart width={200} height={100} data={confidences}>
          <XAxis dataKey="label" tick={{ fontSize: 15 }} />
          <Tooltip />
          <Bar dataKey="confidence" nameKey="label" fill="#f62252" />
        </BarChart>
      </Box>
    )
  }
}
