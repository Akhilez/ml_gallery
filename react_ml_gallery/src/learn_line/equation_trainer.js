import {FlexboxGrid, Input} from 'rsuite';
import React from "react";
import {LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line} from 'recharts';
import './learn_line.css';
import '../commons/components/components.css';
import MLHelper from "./neural_net";
import * as d3 from "d3";


export default class EquationTrainer extends React.Component {
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
        this.d3Bridge = new D3Bridge();
    }

    render() {
        return (
            <div>
                <Neuron bridge={this.d3Bridge}/>
                {/*this.getNeuron()*/}
                <h3>Learn from equation</h3>
                <p>Set "m" and "c" values and train the Neural Network to predict these values.</p>
                {this.getEquationInput()}
                <button className={"ActionButton"} onClick={() => this.startTrainingPipeline()}>TRAIN</button>
                {this.state.didTrainingStart && this.showStopTrainingButton()}
                {this.showTrainingData()}
                {this.state.didTrainingStart && `Predicted (m, c) = (${this.state.predM}, ${this.state.predC})`}
                {this.state.didTrainingStart && this.getGraph()}
                {this.state.didTrainingStart && this.getLossGraph()}
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

        this.d3Bridge.d3Component.update({radius: 50});
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

                    self.showLoss(loss, t++);
                    self.updatePredLine();

                    trainingLoop(epoch + 1);

                }, 100);  // wait 5000 milliseconds then recheck
            }
        };

        trainingLoop(0);

    }

    getNeuron() {
        return (
            <svg width={"500px"} height={"300px"}>
                <circle cx={""}/>
                <line />

                <circle />
                <line />

                <circle cx={50} cy={50} r={10} fill="red"/>
                <line/>
            </svg>
        );
    }

    getEquationInput() {
        return (
            <div style={{fontSize: 40}}>
                <FlexboxGrid>
                    <FlexboxGrid.Item>y = </FlexboxGrid.Item>
                    <FlexboxGrid.Item>
                        {this.getParamsPicker("M")}
                    </FlexboxGrid.Item>
                    <FlexboxGrid.Item> x + </FlexboxGrid.Item>
                    <FlexboxGrid.Item>
                        {this.getParamsPicker("C")}
                    </FlexboxGrid.Item>
                </FlexboxGrid>
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

    updatePredLine() {
        let data = [];
        let predParams = this.nn.getWeights();
        for (let i = 0; i < this.state.data.length; i++) {
            let column = this.state.data[i];
            column.predX = column.realX;
            column.predY = parseFloat(column.realX) * parseFloat(predParams.m[0]) + parseFloat(predParams.c[0]);
            data.push(column);
        }
        this.setState({data: data, predM: predParams.m, predC: predParams.c});
    }

    getGraph() {
        return (
            <div>
                <LineChart
                    width={500}
                    height={300}
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

    showTrainingData() {
        if (this.state.randomX.length === 0)
            return (<p>Change the values of M and C and click on TRAIN button.</p>);
        return (
            <FlexboxGrid>
                <FlexboxGrid.Item>Training Data: </FlexboxGrid.Item>
                {
                    this.state.randomX.map((value, index) => {
                        return (
                            <FlexboxGrid.Item style={{margin: 3}} key={index}>
                                ({parseFloat(this.state.randomX[index]).toFixed(2)}, {parseFloat(this.state.randomY[index]).toFixed(2)})
                            </FlexboxGrid.Item>
                        );
                    })
                }
            </FlexboxGrid>
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

class D3Bridge {
    constructor() {
        this.d3Component = null;
        this.update = (data) => {
        };
    }
}

class Neuron extends React.Component {

    constructor(props) {
        super(props);
        this.svg = null;
        this.bridge = props.bridge;
        props.bridge.d3Component = this;
        this.bridge.update = this.update;
        this.data = [{radius: 10}];
    }

    update(data) {
        console.log("Updating the svg");
        console.log(data);
        this.data = [data];
        this.svg.data(this.data)
            .enter()
            .attr("r", 100);
    }

    build() {
        console.log("data: ");
        console.log(this.data);
        this.svg.data(this.data)
            .append("circle")
            .attr("cx", 150)
            .attr("cy", 70)
            .attr("r", 6);
    }

    componentDidMount() {
        // D3 Code to create the chart
        // using this._rootNode as container

        let container = d3.select(this._rootNode);
        this.svg = container.append('svg');
    }

    shouldComponentUpdate(nextProps, nextState, nextContext) {
        // Prevents component re-rendering
        return false;
    }

    _setRef(componentNode) {
        this._rootNode = componentNode;
    }

    render() {
        return (
            <div ref={this._setRef.bind(this)}/>
        );
    }
}
