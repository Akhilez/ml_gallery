import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";
import MnistClassifier from "./mnist_classifier";
import NumberPaintCanvas from "./paint_canvas";


export default class WhichCharPage extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            isTraining: false,
            lossData: [],
        };

        this.paintCanvasRef = React.createRef();
        this.convNet = new MnistClassifier(this);

    }

    render() {
        return (
            <div>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Which Character?</h1>
                        <p>
                            [IN PROGRESS] <br/> Predict which number is being drawn.
                        </p><br/>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                        {!this.state.isTraining &&
                        <button className={"ActionButton"} onClick={() => this.startTraining()}>TRAIN</button>}
                        {this.state.isTraining &&
                        <button className={"PassiveButton"} onClick={() => this.stopTraining()}>STOP</button>}
                        <br/>

                        <NumberPaintCanvas ref={this.paintCanvasRef}/>

                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }
}