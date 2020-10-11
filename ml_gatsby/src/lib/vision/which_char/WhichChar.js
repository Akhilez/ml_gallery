import React from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import MnistClassifier from "./classifier"
import { Centered } from "../../components/commons"
import { MdRefresh } from "react-icons/all"
import NumberPaintCanvas from "./paint_canvas"
import { Box, Button, Flex, IconButton, Text } from "@chakra-ui/core"
import { Bar, BarChart, Tooltip, XAxis } from "recharts"

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
    this.convNet.initialize_data(this.drawSamples)

    this.canvasRef = React.createRef()

    this.sampleRefs = []
    for (let i = 0; i < 10; i++) this.sampleRefs.push(React.createRef())
    this.sampleSide = 100
  }

  drawSamples = () => {
    const sampleData = this.convNet.getSamples(this.sampleSide)
    for (let i = 0; i < 10; i++) {
      const canvas = this.sampleRefs[i].current
      const ctx = canvas.getContext("2d")
      const arr2 = new Uint8ClampedArray(40000) // TODO: Update this to 10 images

      // Initialize a new ImageData object
      let imageData = new ImageData(sampleData[i], this.sampleSide)

      // Draw image data to the canvas
      ctx.putImageData(imageData, 20, 20)
    }
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          {!this.state.modelLoaded && "Loading model..."}

          {this.state.modelLoaded && (
            <>
              <NumberPaintCanvas
                ref={this.paintCanvasRef}
                parent={this}
                mt={6}
              />
              <IconButton
                aria-label="icon"
                icon={MdRefresh}
                isRound
                variant="outline"
                variantColor="red"
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
                  variantColor="brand"
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
                    variantColor="brand"
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
        </Centered>
      </ProjectWrapper>
    )
  }

  Samples = () => {
    return (
      <Flex display="inline-block">
        {this.sampleRefs.map((ref, index) => (
          <canvas
            key={index}
            ref={ref}
            height={`${this.sampleSide}px`}
            width={`${this.sampleSide}px`}
          />
        ))}
      </Flex>
    )
  }

  startTraining() {
    this.setState({ isTraining: true })
    this.convNet.train()
  }

  stopTraining() {
    this.setState({ isTraining: false })
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
