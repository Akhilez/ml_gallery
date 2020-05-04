import {Input} from 'rsuite';
import React from "react";
import {LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line} from 'recharts';
import './learn_line.css';
import '../commons/components/components.css';
import MLHelper from "./neural_net";
import Sketch from "react-p5";


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

        this.neuronRef = React.createRef();
    }

    render() {
        return (
            <div>
                <h3 style={{marginTop: "50px"}}>Learn from equation</h3>
                {<Neuron ref={this.neuronRef}/>}
                <p>Set "m" and "c" values and train the Neural Network to predict these values.</p>
                {this.getEquationInput()}
                <button className={"ActionButton"} onClick={() => this.startTrainingPipeline()}>TRAIN</button>
                {this.state.didTrainingStart && this.showStopTrainingButton()}
                {this.state.didTrainingStart && this.getGraph()}
                {this.state.didTrainingStart && this.getParametersGraph()}
                {this.state.didTrainingStart && this.getLossGraph()}
            </div>
        );
    }

    getParametersGraph() {
        return (
            <div>
                <table className={"table"} style={{width: 300}}>
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
            column.predY = parseFloat(column.realX) * parseFloat(predParams.m[0]) + parseFloat(predParams.c[0]);
            data.push(column);
        }
        this.setState({data: data, predM: predParams.m, predC: predParams.c});
    }

    getGraph() {
        return (
            <div>
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

class Neuron extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            w: 0.1,
            b: 0.1,
        };

        this.height = 300;
        this.width = 600;

        this.cx = this.width / 2;
        this.cy = this.height / 2;

        this.lineLength = 150;

    }

    render() {
        return (
            <Sketch setup={(p5, parent) => this.setup(p5, parent)} draw={p5 => this.draw(p5)}/>
        );
    }

    set(state) {
        this.setState(state);
    }

    setup(p5, parent) {
        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(10);
    }

    draw(p5) {
        p5.background(243);
        p5.textSize(18);

        // Weight
        if (this.state.w < 0) p5.stroke(247, 120, 35);
        else p5.stroke(235, 16, 93);
        p5.strokeWeight(this.rescale(this.state.w));
        p5.line(this.cx, this.cy, this.cx - this.lineLength, this.cy - 50);
        p5.text(`m = ${this.state.w}`, this.cx - 110, this.cy - 50);

        // Bias
        if (this.state.b < 0)
            p5.stroke(247, 120, 35);
        else
            p5.stroke(235, 16, 93);
        p5.strokeWeight(this.rescale(this.state.b));
        p5.line(this.cx, this.cy, this.cx - this.lineLength, this.cy + 50);
        p5.text(`c = ${this.state.b}`, this.cx - 110, this.cy + 60);

        // y
        p5.stroke(100, 100, 100);
        p5.strokeWeight(1);
        p5.line(this.cx, this.cy, this.cx + this.lineLength, this.cy);

        // Circle
        p5.fill(235, 16, 93);
        p5.noStroke();
        p5.ellipse(this.cx, this.cy, 50);

        // x
        p5.text('x', this.cx - this.lineLength - 20, this.cy - 45);

        // 1
        p5.text('1', this.cx - this.lineLength - 20, this.cy + 55);

        // y
        p5.text('y', this.cx + this.lineLength + 20, this.cy + 5);


    }

    rescale(t) {
        return Math.tanh(t);
    }
}
