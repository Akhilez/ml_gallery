import React from "react";
import Neuron from "./neuron";
import Sketch from "react-p5";
import TrailManager from '../commons/trail_manager';


class PointsTrainer extends React.Component {

    constructor(props){
        super(props);
        this.state = {};
        this.neuronRef = React.createRef();
        this.graphRef = React.createRef();
    }

    render() {
        return (
            <div style={{marginTop: 50, marginBottom: 50}}>
                <h2>Learn from points</h2>
                <p>Create points that are approximately linear and train the Neural Network to predict the best line equation.</p>
                <Neuron ref={this.neuronRef}/>
                <Graph ref={this.graphRef}/>
                <button className={"ActionButton"}>TRAIN</button>
            </div>
        );
    }

}


class Graph extends React.Component{

    constructor(props){
        super(props);
        this.state = {
            points: [],
            m: 0,
            c: 0,
            isTraining: false,
        };
        this.trails = new TrailManager();
    }

    render() {
        return (
            <Sketch setup={(p5, parent) => this.setup(p5, parent)} draw={p5 => this.draw(p5)}/>
        );
    }

    setup(p5, parent){
        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(10);
    }

    draw(p5){
        p5.background(200);

        // TODO: Draw the axes.



    }
}
