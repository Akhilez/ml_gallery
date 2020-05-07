import React from "react";
import {Container} from "react-bootstrap";
import ProjectsNav from "../commons/components/projects_nav";
import BreadCrumb from "../commons/components/breadcrumb";
import MLAppBar from "../commons/components/ml_app_bar";
import Sketch from "react-p5";
import TrainingTracker from "../commons/utils/training_tracker";
import Chartist from "../commons/utils/chartist";
import LinearClassifierNeuron from "./linear_classifier_neuron";
import Neuron from "../commons/components/neuron";
import {Centered} from "../commons/components/components";
import ProjectPaginator from "../commons/components/project_paginator";
import '../commons/components/components.css';


export default class LinearClassifierPage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.graphRef = React.createRef();
        this.neuronRef = React.createRef();
    }

    render() {
        return (
            <div>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Linear Classifier</h1>
                        <Neuron ref={this.neuronRef}/>
                        <button className={"ActionButton"} onClick={() => this.graphRef.current.startTraining()}>TRAIN
                        </button>
                        <button className={"ActionButton"} onClick={() => this.graphRef.current.stopTraining()}>STOP
                        </button>
                        <button className={"ActionButton"} onClick={() => this.graphRef.current.removeData()}>CLEAR
                            DATA
                        </button>
                        <Graph ref={this.graphRef} neuronRef={this.neuronRef}/>
                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }
}


class Graph extends React.Component {

    constructor(props) {
        super(props);
        this.neuron = new LinearClassifierNeuron();
        this.state = {
            isTraining: false,
        };

        this.height = 500;
        this.width = 800;

        this.tracker = new TrainingTracker();
        this.chartist = null;

        this.neuronRef = props.neuronRef;
    }

    render() {
        return (
            <div className={"rounded"}>
                <Sketch setup={(p5, parent) => this.setup(p5, parent)} draw={p5 => this.draw(p5)}
                        mouseClicked={(p5) => this.handleInput(p5)}/>
            </div>
        );
    }

    setup(p5, parent) {
        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(this.tracker.frameRate);
        this.chartist = new Chartist(p5, this.width, this.height);
    }

    draw(p5) {
        p5.background(243);

        this.chartist.drawPoints(this.neuron.getDataPoints());
        let params = this.neuron.getMC();
        this.chartist.drawLine(params.w, params.b);

        if (this.tracker.isComplete() || !this.state.isTraining)
            return;

        this.tracker.updateFrame();

        if (this.tracker.isNewEpoch()) {
            this.neuron.fullPass();
            this.neuronRef.current.set({w: params.w, b: params.b});
        }

    }

    handleInput(p5) {
        let label = 1;
        if (p5.keyIsDown(p5.SHIFT))
            label = 0;
        let x = p5.mouseX / this.width;
        let y = p5.mouseY / this.height;
        this.neuron.addDataPoint(x, y, label);
    }

    startTraining() {
        this.setState({isTraining: true});
        this.tracker.epoch = 0;
    }

    stopTraining() {
        this.setState({isTraining: false});
    }

    removeData() {
        this.neuron.removeData();
    }

}
