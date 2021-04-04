import React, { Fragment } from "react"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import { projects } from "../../globals/data"
import IrisNet from "./IrisNet"
import NeuralGraph from "./NeuralGraph"
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
import {
  Box,
  Button,
  Flex,
  Slider,
  SliderTrack,
  SliderFilledTrack,
  SliderThumb,
  Text,
  Stack,
  Heading,
} from "@chakra-ui/react"
import { DynamicColorBox } from "../../components/dynamicColorMode"

export class DeepIris extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.deep_iris
    this.state = {
      nNeurons: [6, 4],
      isTraining: false,
      lossData: [],
      petalWidth: (Math.random() + 0.5) * 50,
      petalHeight: (Math.random() + 0.5) * 50,
      sepalWidth: (Math.random() + 0.5) * 50,
      sepalHeight: (Math.random() + 0.5) * 50,
      confidences: [0, 0, 0],
    }

    this.graphRef = React.createRef()
    this.irisNet = new IrisNet(this)
  }
  render() {
    return (
      <ProjectWrapper project={this.project} align="center">
        <Flex
          justifyContent={{ md: "center" }}
          alignItems="center"
          overflow="auto"
        >
          <this.Sliders />
          <NeuralGraph
            ref={this.graphRef}
            state={this.state}
            actions={{
              updateNeurons: (layerNumber, change) =>
                this.updateNeurons(layerNumber, change),
            }}
          />
          <this.PredictionChart />
        </Flex>

        {this.UpdateLayersButtons()}

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

        <this.Samples />

        {this.getLossGraph()}
      </ProjectWrapper>
    )
  }

  Samples = () => {
    return (
      <Box>
        <Text mt={6} mb={4}>
          Click a sample class below to copy
        </Text>
        <Flex justifyContent={{ md: "center" }} overflow="auto" pb={8} px={4}>
          <this.SampleBox values={[20.24, 92.8, 9.24, 9.84]} cls="Satosa" />
          <this.SampleBox values={[57.44, 27, 65.2, 53.04]} cls="Versicolor" />
          <this.SampleBox
            values={[83.52, 47.4, 91.04, 81.04]}
            cls="Virginica"
          />
        </Flex>
      </Box>
    )
  }

  SampleBox = ({ values, cls }) => {
    const texts = ["Sepal Height", "Sepal Width", "Petal Height", "Petal Width"]
    return (
      <DynamicColorBox
        width="200px"
        minW="150px"
        p={4}
        borderRadius="15px"
        mx={2}
        className="ProjectContainer"
        boxShadow="lg"
        onClick={() =>
          this.setState({
            sepalHeight: values[0],
            sepalWidth: values[1],
            petalHeight: values[2],
            petalWidth: values[3],
          })
        }
        role="group"
        _hover={{
          backgroundColor: "secondary.500",
          transitionDuration: "0.4s",
          color: "white",
        }}
      >
        <Box>
          <Heading
            as="h2"
            fontSize="xl"
            mb={6}
            _groupHover={{ color: "white", fontWeight: "bold" }}
          >
            {cls}
          </Heading>
          {values.map((value, index) => (
            <Fragment key={index}>
              <Text
                textAlign="left"
                fontSize="md"
                mt={4}
                _groupHover={{ color: "white" }}
              >
                {texts[index]}
              </Text>
              <Slider value={value} color="brand" isDisabled max={100} min={0}>
                <SliderTrack />
                <SliderFilledTrack />
                <SliderThumb />
              </Slider>
            </Fragment>
          ))}
        </Box>
        {this?.props?.children}
      </DynamicColorBox>
    )
  }

  getLossGraph() {
    return (
      <Box maxW="500px" mt={8}>
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
    this.graphRef.current.neuronUpdateClickActions = []
  }

  UpdateLayersButtons() {
    return (
      <Box my={4}>
        Change Depth:
        <Button
          variant="outline"
          colorScheme="brand"
          ml={2}
          size="sm"
          isDisabled={this.state.isTraining}
          onClick={() => this.updateLayers(1)}
        >
          +
        </Button>
        <Button
          variant="outline"
          colorScheme="brand"
          ml={2}
          size="sm"
          isDisabled={this.state.isTraining}
          onClick={() => this.updateLayers(-1)}
        >
          -
        </Button>
      </Box>
    )
  }

  PredictionChart = () => {
    let confidences = this.irisNet.predict([
      [
        this.state.sepalHeight,
        this.state.sepalWidth,
        this.state.petalHeight,
        this.state.petalWidth,
      ],
    ])
    const classes = ["Setosa", "Versicolor", "Virginica"]
    confidences = confidences.map((confidence, index) => {
      return { confidence, cls: classes[index] }
    })

    return (
      <Box>
        <BarChart width={200} height={100} data={confidences} layout="vertical">
          <XAxis type="number" hide />
          <YAxis type="category" dataKey="cls" tick={{ fontSize: 12 }} />
          <Tooltip />
          <Bar dataKey="confidence" nameKey="cls" fill="#f62252" />
        </BarChart>
      </Box>
    )
  }

  Sliders = () => {
    return (
      <Box minW="150px" mr={4}>
        <Text textAlign="left" fontSize="sm">
          Sepal Height
        </Text>

        <Slider
          mb={4}
          value={this.state.sepalHeight}
          color="brand"
          onChange={value => this.setState({ sepalHeight: value })}
        >
          <SliderTrack />
          <SliderFilledTrack />
          <SliderThumb />
        </Slider>

        <Text textAlign="left" fontSize="sm">
          Sepal Width
        </Text>

        <Slider
          mb={4}
          value={this.state.sepalWidth}
          color="brand"
          onChange={value => this.setState({ sepalWidth: value })}
        >
          <SliderTrack />
          <SliderFilledTrack />
          <SliderThumb />
        </Slider>

        <Text textAlign="left" fontSize="sm">
          Petal Height
        </Text>

        <Slider
          mb={4}
          value={this.state.petalHeight}
          color="brand"
          onChange={value => this.setState({ petalHeight: value })}
        >
          <SliderTrack />
          <SliderFilledTrack />
          <SliderThumb />
        </Slider>

        <Text textAlign="left" fontSize="sm">
          Petal Width
        </Text>

        <Slider
          mb={4}
          value={this.state.petalWidth}
          color="brand"
          onChange={value => this.setState({ petalWidth: value })}
        >
          <SliderTrack />
          <SliderFilledTrack />
          <SliderThumb />
        </Slider>
      </Box>
    )
  }
}
