import React from "react";
import {Container} from "react-bootstrap";
import ProjectsNav from "../commons/components/projects_nav";
import BreadCrumb from "../commons/components/breadcrumb";
import MLAppBar from "../commons/components/ml_app_bar";
import Sketch from "react-p5";
import TrainingTracker from "../commons/utils/training_tracker";
import Chartist from "../commons/utils/chartist";
import LinearClassifierNeuron from "./linear_classifier_neuron";


export default class LinearClassifierPage extends React.Component {
    constructor(props) {
        super(props);
        this.state = {};
        this.graphRef = React.createRef();
    }

    render() {
        return (
            <div>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <h1>Linear Classifier</h1>
                    <Graph ref={this.graphRef}/>
                </Container>
            </div>
        );
    }
}


class Graph extends React.Component{

    constructor(props){
        super(props);
        this.neuron = new LinearClassifierNeuron();
        this.state = {
            points: this.neuron.getDataPoints(),
            m: 0,
            c: 0,
        };

        this.height = 500;
        this.width = 800;

        this.tracker = new TrainingTracker();
        this.chartist = null;
    }

    render() {
        return (
            <Sketch setup={(p5, parent) => this.setup(p5, parent)} draw={p5 => this.draw(p5)}/>
        );
    }

    setup(p5, parent){
        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(this.tracker.frameRate);
        this.chartist = new Chartist(p5, this.width, this.height);
    }

    draw(p5){
        p5.background(200);

        this.chartist.drawPoints(this.neuron.getDataPoints());
        let params = this.neuron.getMC();
        this.chartist.drawLine(params.m, params.c);

        if (this.tracker.isComplete())
            return;

        this.tracker.updateFrame();

        // TODO: Get mouse input


    }

}
