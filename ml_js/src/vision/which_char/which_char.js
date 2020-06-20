import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";
import MnistClassifier from "./mnist_classifier";
import NumberPaintCanvas from "./paint_canvas";
import {HOST} from "../../commons/settings";
import { IconButton } from '@material-ui/core';
import { Refresh } from '@material-ui/icons';


export default class WhichCharPage extends React.Component {

    constructor(props) {
        super(props);

        this.state = {
            isTraining: false,
            lossData: [],
            modelLoaded: false,
            predicted: null,
        };

        this.paintCanvasRef = React.createRef();
        this.convNet = new MnistClassifier(this);

    }

    componentDidMount() {
        this.convNet.initialize_model();
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
                            Predict which number is being drawn.
                        </p><br/>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>
                        {!this.state.modelLoaded && <>Loading model...<br/></>}

                        {this.state.modelLoaded &&
                        <>
                            <NumberPaintCanvas ref={this.paintCanvasRef} parent={this}/>
                            <div>
                                <Refresh onClick={()=>this.paintCanvasRef.current.clearCanvas()}/><br/>
                                Predicted: {this.state.predicted}
                            </div>
                        </>
                        }

                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </div>
        );
    }

    startTraining() {
    }

    stopTraining() {
    }
}