import React from "react";
import {Container} from "react-bootstrap";
import MLAppBar from "../commons/ml_app_bar";
import {Centered, OutlinedButtonLink} from "../commons/components/components";
import '../landing/landing.css';
import './learn_line.css';
import '../commons/components/components.css';
import ProjectsNav from "../commons/components/projects_nav";
import BreadCrumb from "../commons/components/breadcrumb";
import MLHelper from "./neural_net";
import Neuron from "./neuron";
import ProjectPaginator from "../commons/components/project_paginator";
import {CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis} from "recharts";
import {Input} from "rsuite";


export default class LearnLinePage extends React.Component {
    render() {
        return (
            <div>
                <div style={{float: "left"}}>
                    <ProjectsNav activeKey={this.props.project.id}/>
                </div>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Learn A Line</h1>
                        <p>Predict the m and c values of the straight line (y = mx + c) equation.</p>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>
                        <EquationTrainer/>
                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }
}

class EquationTrainer extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            m: null,
            c: null,
            randomX: [],
            randomY: [],
            predM: null,
            predC: null,
            data: [],
            lossData: [],
            didTrainingStart: false,
            isTraining: false,
        };
        this.nn = new MLHelper();

        this.neuronRef = React.createRef();
    }

    render() {
        return (
            <div>
                <Neuron ref={this.neuronRef}/>
                <p>Set "m" and "c" values and train the Neural Network to predict these values.</p>
                {this.getEquationInput()}
                <button className={"ActionButton"} onClick={() => this.startTrainingPipeline()}>TRAIN</button>
                {this.state.didTrainingStart && this.showStopTrainingButton()}
                {this.state.didTrainingStart && this.getGraph()}
                <div>
                    <div className={"inline"}>
                        {this.state.didTrainingStart && this.getParametersGraph()}
                    </div>
                    <div className={"inline"}>
                        {this.state.didTrainingStart && this.getLossGraph()}
                    </div>
                </div>
            </div>
        );
    }

    getParametersGraph() {
        return (
            <div>
                <table className={"table"} style={{width: 300}}>
                    <tbody>
                    <tr>
                        <th/>
                        <th>Real</th>
                        <th>Predicted</th>
                    </tr>
                    <tr>
                        <td>m</td>
                        <td>{this.state.m}</td>
                        <td>{this.state.predM}</td>
                    </tr>
                    <tr>
                        <td>c</td>
                        <td>{this.state.c}</td>
                        <td>{this.state.predC}</td>
                    </tr>
                    </tbody>
                </table>
            </div>
        );
    }

    startTrainingPipeline() {

        if (this.state.m == null) {
            this.showError("M is not set.");
            return;
        }
        if (this.state.c == null) {
            this.showError("C is not set.");
            return;
        }

        let m = this.state.m;
        let c = this.state.c;

        let randomData = this.nn.generateRandomLineData(m, c, 10);

        this.setState({
            realM: m,
            realC: c,
            randomX: randomData.x,
            randomY: randomData.y,
            isTraining: true,
            didTrainingStart: true,
            lossData: [],
            data: this.createRealData(randomData.x, randomData.y),
        }, () => {
            this.train(randomData.x, randomData.y);
        });

    }

    train(x, y) {
        let epochs = 1000;
        let t = 0;
        let self = this;

        let trainingLoop = function (epoch) {
            if (epoch <= epochs) {
                setTimeout(() => {

                    let randomIndex = Math.floor(Math.random() * x.length);

                    if (!self.state.isTraining) {
                        epoch = epochs;
                        return
                    }
                    let loss = self.nn.fullPass(x[randomIndex], y[randomIndex]);

                    let predParams = self.nn.getWeights();

                    self.showLoss(loss, t++);
                    self.updatePredLine(predParams);
                    self.neuronRef.current.set({w: predParams.m, b: predParams.c});

                    trainingLoop(epoch + 1);

                }, 100);  // wait 5000 milliseconds then recheck
            }
        };

        trainingLoop(0);

    }

    getEquationInput() {
        return (
            <div style={{fontSize: 40}}>
                <div className={"inline"}>y =</div>
                <div className={"inline"}>{this.getParamsPicker("M")}</div>
                <div className={"inline"}> x +</div>
                <div className={"inline"}>{this.getParamsPicker("C")}</div>
            </div>
        );
    }

    createRealData(x, y) {
        let data = [];
        for (let i = 0; i < x.length; i++) {
            data.push({realX: x[i], realY: y[i]});
        }
        return data;
    }

    showLoss(loss, index) {
        this.setState({lossData: this.state.lossData.concat([{index: index, loss: loss.dataSync()[0]}])});
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

    updatePredLine(predParams) {
        let data = [];
        for (let i = 0; i < this.state.data.length; i++) {
            let column = this.state.data[i];
            column.predX = column.realX;
            column.predY = parseFloat(column.realX) * parseFloat(predParams.m) + parseFloat(predParams.c);
            data.push(column);
        }
        this.setState({data: data, predM: predParams.m, predC: predParams.c});
    }

    getGraph() {
        return (
            <div style={{marginTop: 100, marginBottom: 100}}>
                <LineChart
                    width={800}
                    height={500}
                    data={this.state.data}
                    margin={{
                        top: 5, right: 30, left: 20, bottom: 5,
                    }}
                >
                    <CartesianGrid strokeDasharray="3 3"/>
                    <XAxis dataKey="realX" type="number" scale="auto"/>
                    <YAxis/>
                    <Tooltip/>
                    <Legend/>
                    <Line type="monotone" dataKey="realY" stroke="#8884d8"/>
                    <Line type="monotone" dataKey="predY" stroke="#82ca9d"/>
                </LineChart>
            </div>
        );
    }

    showStopTrainingButton() {
        return (
            <button className={"ActionButton"} onClick={() => this.setState({isTraining: false})}>Stop</button>
        );
    }

    showError(message) {
        alert(message);
    }

    getParamsPicker(params) {
        if (params === "M")
            return (
                <Input className={"inputBox"} placeholder="m" type={"number"} onChange={(value, event) => {
                    this.setState({m: value})
                }}/>
            );
        else if (params === "C") {
            return (
                <Input className={"inputBox"} placeholder="c" type={"number"} onChange={(value, event) => {
                    this.setState({c: value})
                }}/>
            );
        }
    }
}
