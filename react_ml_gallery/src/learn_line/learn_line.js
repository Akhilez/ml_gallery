import React from "react";
import {Container as BootstrapContainer} from "react-bootstrap";
import MLAppBar from "../commons/ml_app_bar";
import {Centered} from "../commons/components/components";
import '../landing/landing.css';
import ProjectsNav from "../commons/components/projects_nav";
import {Button, Container, Sidebar} from "rsuite";


export default class LearnLinePage extends React.Component {
    render() {
        return (
            <div>
                <div style={{float: "left"}}>
                    <ProjectsNav/>
                </div>
                <BootstrapContainer>
                    <MLAppBar/>
                    <Centered>
                        <h1>Learn A Line</h1>
                        <p>Predict the m and c values of the straight line (y = mx + c) equation.</p>
                    </Centered>
                </BootstrapContainer>
            </div>
        );
    }
}

