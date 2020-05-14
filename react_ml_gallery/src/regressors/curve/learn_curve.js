import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";
import {MLPyHost, MLPyPort} from "../../commons/settings";


export default class LearnCurvePage extends React.Component {

    constructor(props) {
        super(props);
        this.mlPyUrl = `ws://${MLPyHost}:${MLPyPort}/ws/poly_reg`;
        this.traceId = null;
        this.socket = new WebSocket(this.mlPyUrl);
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
                        <button className={"ActionButton"} onClick={() => this.startTraining()}>TRAIN</button>
                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }

    startTraining() {
        let payload = {
            action: 'start_training',
            trace_id: this.traceId,
        };
        this.socket.send(JSON.stringify(payload));
    }

    setupSocketListeners() {
        this.socket.onclose = () => {
            console.error('Chat socket closed unexpectedly')
        };
        this.socket.onmessage((event) => {
            const data = JSON.parse(event.data);
            this.traceId = data.trace_id;
            if (data.action === 'status_update') {
                this.updateTrainingStatus(data.data);
            }
        });
    }

    updateTrainingStatus(data) {

    }
}
