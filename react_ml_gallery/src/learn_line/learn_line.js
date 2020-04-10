import React from "react";
import {Container} from "react-bootstrap";
import MLAppBar from "../commons/ml_app_bar";
import {Centered, OutlinedButtonLink} from "../commons/components/components";
import '../landing/landing.css';
import ProjectsNav from "../commons/components/projects_nav";
import BreadCrumb from "../commons/components/breadcrumb";
import EquationTrainer from "./equation_trainer";
import PointsTrainer from "./points_trainer";


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
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/>
                    </Centered>
                    <EquationTrainer/>
                    <PointsTrainer/>
                </Container>
            </div>
        );
    }
}

