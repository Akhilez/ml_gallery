import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";
import {MLPyHost, MLPyPort} from "../../commons/settings";
import '../../commons/components/components.css';
import Graph from './sketch_learn_curve';
import NeuronGraphLearnCurve from "./neuron_graph_learn_curve";
import {CartesianGrid, Legend, Line, LineChart, Tooltip, XAxis, YAxis} from "recharts";


export default class LearnCurvePage extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            order: 5,
            isTraining: false,
            loss: null,
            lossData: [],
            isTrainerInitialized: false,
        };

        this.mlPyUrl = `ws://${MLPyHost}:${MLPyPort}/ws/poly_reg`;
        this.traceId = null;
        this.socket = new WebSocket(this.mlPyUrl);
        this.x = null;
        this.y = null;

        this.graphRef = React.createRef();
        this.neuronRef = React.createRef();

        // new WebSocket('ws://py.ml.akhilez.com:8000/ws/poly_reg');

    }

    componentDidMount() {
        this.setupSocketListeners();
    }

    render() {
        return (
            <div>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Learn A Curve</h1>
                        <p>
                            A single neuron can approximate any continuous polynomial function in 2D space.<br/>
                            Here, you can train a single neuron to fit the your own data.<br/>
                            Click on the canvas below ([-1, 1] graph) to create a point in the 2D space.<br/>
                            For best results, create your points in a curvy pattern.
                        </p><br/>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                        {!this.state.isTrainerInitialized && <p>Connecting to webserver ...</p>}

                        <NeuronGraphLearnCurve ref={this.neuronRef}/>

                        {this.state.isTrainerInitialized &&
                        <button className={"ActionButton"} onClick={() => this.startTraining()}>TRAIN</button>}
                        {this.state.isTrainerInitialized &&
                        <button className={"PassiveButton"} onClick={() => this.stopTraining()}>STOP</button>}
                        {this.state.isTrainerInitialized &&
                        <button className={"PassiveButton"} onClick={() => this.clearData()}>CLEAR</button>}

                        {this.state.isTrainerInitialized && this.getComplexityModifier()}

                        <br/>
                        <Graph ref={this.graphRef} new_point_classback={(x, y) => this.add_new_point(x, y)}/>
                        {this.state.isTrainerInitialized && this.getLossGraph()}
                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }

    getComplexityModifier() {

        let terms = ["y = "];

        for (let i = 1; i <= this.state.order; i++) {
            terms.push(<div className={"inline"} key={`eqn-${i}`}>w<sub>{i}</sub>x<sup>{this.state.order - i + 1}</sup> + &nbsp;</div>);
        }
        terms.push("b");

        return (
            <div style={{fontSize: 20, marginTop: 50}}>
                <div style={{fontSize: 28, marginBottom: 20}}>
                    {terms.map((item) => item)} <br/>
                </div>
                Change Complexity:
                <button className={"PassiveButton"} onClick={() => this.changeOrder(1)}>+</button>
                <button className={"PassiveButton"} onClick={() => this.changeOrder(-1)}>-</button>
            </div>
        );
    }

    startTraining() {
        let payload = {
            action: 'start_training',
            trace_id: this.traceId,
        };
        console.log(JSON.stringify(payload));
        this.socket.send(JSON.stringify(payload));
        this.setState({isTraining: true});
    }

    stopTraining() {
        let payload = {
            action: 'stop_training',
            trace_id: this.traceId,
        };
        this.socket.send(JSON.stringify(payload));
        this.setState({isTraining: false});
    }

    setupSocketListeners() {
        this.socket.onclose = () => {
            console.error('Chat socket closed unexpectedly')
        };
        this.socket.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data);
            this.traceId = data.trace_id;
            if (data.action === 'status_update') {
                this.updateTrainingStatus(data.data);
            } else if (data.action === 'init') {
                this.x = data.data[0];
                this.y = data.data[1];
                this.traceId = data.trace_id;
                this.setState({isTrainerInitialized: true});
                this.drawDataToCanvas(this.x, this.y);
            }
        };
    }

    updateTrainingStatus(data) {
        this.setState({
            loss: data.train_error,
            lossData: this.state.lossData.concat([{index: this.state.lossData.length, loss: data.train_error}])
        });
        this.graphRef.current.weights = data.weights;
        this.neuronRef.current.weights = data.weights;
    }

    changeOrder(change) {
        if (this.state.order <= 1) return;

        let newOrder = this.state.order + change;

        this.setState({order: newOrder});

        this.neuronRef.current.initializeWeights(newOrder);

        this.socket.send(JSON.stringify({
            action: 'change_order',
            order: newOrder,
            trace_id: this.traceId,
        }));
    }

    clearData() {
        this.graphRef.current.x = [];
        this.graphRef.current.y = [];

        this.socket.send(JSON.stringify({
            action: 'clear_data',
            trace_id: this.traceId,
        }));
    }

    add_new_point(x, y) {
        this.socket.send(JSON.stringify({
            action: 'new_point',
            trace_id: this.traceId,
            x: x,
            y: y,
        }))
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

    drawDataToCanvas(x, y) {
        let newX = [];
        let newY = [];
        for (let i = 0; i < x.length; i++) {
            let [xi, yi] = this.graphRef.current.coordinatesToLengths(x[i], y[i]);
            newX.push(xi);
            newY.push(yi);
        }
        this.graphRef.current.x = newX;
        this.graphRef.current.y = newY;
    }

}
