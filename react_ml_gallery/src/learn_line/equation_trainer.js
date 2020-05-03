import {Input} from 'rsuite';
import React from "react";
import {LineChart, CartesianGrid, XAxis, YAxis, Tooltip, Legend, Line} from 'recharts';
import './learn_line.css';
import '../commons/components/components.css';
import MLHelper from "./neural_net";


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
        this.neuronContext = null;
    }

    render() {
        return (
            <div>
                <h3 style={{marginTop: "50px"}}>Learn from equation</h3>
                {this.getNeuron()}
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

    componentDidMount() {
        console.log(this.neuronRef.current);
        this.neuronContext = this.neuronRef.current.getContext("2d");
        this.drawNeuron()
    }

    getNeuron() {
        return (
            <div>
                <canvas ref={this.neuronRef} width={500} height={500} />
            </div>
        );
    }

    drawNeuron(){
        const ctx = this.neuronContext;
        ctx.beginPath();
        ctx.arc(250, 250, 50, 0, 2 * Math.PI, false);
        ctx.fillStyle = 'green';
        ctx.fill();
        ctx.lineWidth = 5;
        ctx.strokeStyle = '#003300';
        ctx.stroke();
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

                    self.showLoss(loss, t++);
                    self.updatePredLine();

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
