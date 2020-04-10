import {FlexboxGrid, Input} from 'rsuite';
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
            realM: null,
            realC: null,
            data: [
                {
                    name: 'Page A', uv: 4000, pv: 2400, amt: 2400,
                },
                {
                    name: 'Page B', uv: 3000, pv: 1398, amt: 2210,
                },
                {
                    name: 'Page C', uv: 2000, pv: 9800, amt: 2290,
                },
                {
                    name: 'Page D', uv: 2780, pv: 3908, amt: 2000,
                },
                {
                    name: 'Page E', uv: 1890, pv: 4800, amt: 2181,
                },
                {
                    name: 'Page F', uv: 2390, pv: 3800, amt: 2500,
                },
                {
                    name: 'Page G', uv: 3490, pv: 4300, amt: 2100,
                },
            ],
        };
        this.nn = new MLHelper();
    }


    render() {
        return (
            <div style={{marginTop: 50, marginBottom: 50}}>
                <h3>Learn from equation</h3>
                <p>Set "m" and "c" values and train the Neural Network to predict these values.</p>
                <div style={{fontSize: 40}}>
                    <FlexboxGrid>
                        <FlexboxGrid.Item>y = </FlexboxGrid.Item>
                        <FlexboxGrid.Item>
                            {this.getParamsPicker("M")}
                        </FlexboxGrid.Item>
                        <FlexboxGrid.Item> x + </FlexboxGrid.Item>
                        <FlexboxGrid.Item>
                            {this.getParamsPicker("C")}
                            {console.log(this.state.c)}
                        </FlexboxGrid.Item>
                    </FlexboxGrid>
                </div>
                <button className={"ActionButton"} onClick={() => this.startTrainingPipeline()}>TRAIN</button>
                {this.showTrainingData()}
                {this.getGraph()}
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
        });

        this.train(randomData.x, randomData.y);

    }

    train(x, y) {
        for (let i = 0; i < x.length; i++) {
            let loss = this.nn.fullPass(x[i], y[i]);
            this.showLoss(loss, i);
            this.updatePredLine();
        }
    }

    showLoss(loss, index) {
        // TODO: Show loss
    }

    updatePredLine(){
        // TODO: Update prediction line
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
                    <XAxis dataKey="name"/>
                    <YAxis/>
                    <Tooltip/>
                    <Legend/>
                    <Line type="monotone" dataKey="pv" stroke="#8884d8" activeDot={{r: 8}}/>
                    <Line type="monotone" dataKey="uv" stroke="#82ca9d"/>
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
                    console.log("Changing c");
                    console.log(value);
                    this.setState({c: value})
                }}/>
            );
        }
    }
}