import React from "react";
import './landing.css';
import underConstructionImage from './images/under_construction.png';
import MLAppBar from "../commons/components/ml_app_bar";
import {Centered, OutlinedButtonLink} from "../commons/components/components";
import ProjectsNav from "../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import BreadCrumb from "../commons/components/breadcrumb";


export default function ComingSoon(props) {
    return (
        <div>
            <ProjectsNav activeKey={props.project.id}/>
            <Container>
                <MLAppBar/>
                <BreadCrumb path={props.project.links.app}/>
                <Centered>
                    <h1>{props.project.title}</h1>
                    <p>{props.project.desc}</p>
                    <br/><br/>
                    <h1>{"{ Coming Soon }"}</h1>
                    <img src={underConstructionImage} alt={"Under Construction."}/>
                </Centered>
            </Container>
        </div>
    );
}


