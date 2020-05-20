import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";
import {MLPyHost, MLPyPort} from "../../commons/settings";
import '../../commons/components/components.css';
import Graph from './sketch_learn_cure';


export default class LearnCurvePage extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            order: 5,
            isTraining: false,
            loss: null,
            isTrainerInitialized: false,
        };

        this.mlPyUrl = `ws://${MLPyHost}:${MLPyPort}/ws/poly_reg`;
        this.traceId = null;
        this.socket = new WebSocket(this.mlPyUrl);
        this.x = null;
        this.y = null;

        this.graphRef = React.createRef();

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
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                        {this.state.isTrainerInitialized &&
                        <button className={"ActionButton"} onClick={() => this.startTraining()}>TRAIN</button>}
                        {this.state.isTraining &&
                        <button className={"PassiveButton"} onClick={() => this.stopTraining()}>STOP</button>}
                        {this.state.isTrainerInitialized &&
                        <button className={"PassiveButton"} onClick={() => this.clearData()}>CLEAR</button>}

                        <br/>
                        loss: {this.state.loss}
                        {this.getOrderChanger()}
                        <Graph ref={this.graphRef} new_point_classback={(x, y) => this.add_new_point(x, y)}/>
                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }

    getOrderChanger() {
        return (
            <div>
                Order: {this.state.order}
                <button onClick={() => this.changeOrder(1)}>+</button>
                <button onClick={() => this.changeOrder(-1)}>-</button>
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
        this.setState({loss: data.train_error});
    }

    changeOrder(change) {
        if (this.state.order <= 0) return;

        let newOrder = this.state.order + change;

        this.setState({order: newOrder});

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
