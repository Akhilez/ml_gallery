import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import NeuralGraphIris from "./neural_graph_iris";
import ProjectPaginator from "../../commons/components/project_paginator";
import {CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis} from "recharts";
import IrisNet from './iris_net';


export default class DeepIrisPage extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            nNeurons: [6, 4],
            isTraining: false,
            lossData: [],
        };

        this.graphRef = React.createRef();
        this.irisNet = new IrisNet(this);

    }


    render() {
        return (
            <div>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Deep Iris</h1>
                        <p>
                            [IN PROGRESS] <br/> Predict the type of the flower based on its petal and sepal length.
                        </p><br/>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                        <NeuralGraphIris ref={this.graphRef} appState={this.state} actions={{
                            updateNeurons: (layerNumber, change) => this.updateNeurons(layerNumber, change)
                        }}/>

                        {this.UpdateLayersButtons()}

                        {!this.state.isTraining &&
                        <button className={"ActionButton"} onClick={() => this.startTraining()}>TRAIN</button>}
                        {this.state.isTraining &&
                        <button className={"PassiveButton"} onClick={() => this.stopTraining()}>STOP</button>}

                        <br/>

                        {this.getLossGraph()}

                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }

    getLossGraph() {
        return (
            <LineChart
                width={500}
                height={300}
                data={this.state.lossData}
                margin={{
                    top: 5, right: 30, left: 20, bottom: 5,
                }}
            >
                <CartesianGrid strokeDasharray="3 3"/>
                <XAxis dataKey="index" type="number" scale="auto"/>
                <YAxis/>
                <Tooltip/>
                <Legend/>
                <Line type="monotone" dataKey="loss" stroke="#8884d8"/>
            </LineChart>
        );
    }

    startTraining() {
        this.setState({isTraining: true});
        this.irisNet.train();
    }

    stopTraining() {
        this.setState({isTraining: false});
    }

    updateNeurons(layerNumber, change) {
        this.state.nNeurons[layerNumber] += change;
        this.setState({nNeurons: this.state.nNeurons});
        this.irisNet.initialize_net();
    }

    updateLayers(change) {
        if (change > 0) {
            this.state.nNeurons.push(3);
            this.setState({nNeurons: this.state.nNeurons});
        } else {
            this.state.nNeurons.pop();
            this.setState({nNeurons: this.state.nNeurons});
        }
        this.irisNet.initialize_net();
    }

    UpdateLayersButtons() {
        return (
            <div>
                Change Depth:
                <button className={"PassiveButton"} onClick={() => this.updateLayers(1)}>+</button>
                <button className={"PassiveButton"} onClick={() => this.updateLayers(-1)}>-</button>
            </div>
        );
    }

}