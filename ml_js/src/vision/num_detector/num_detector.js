import React from "react";
import ProjectsNav from "../../commons/components/projects_nav";
import {Container} from "react-bootstrap";
import MLAppBar from "../../commons/components/ml_app_bar";
import BreadCrumb from "../../commons/components/breadcrumb";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import ProjectPaginator from "../../commons/components/project_paginator";



export default class NumberDetectorPage extends React.Component {
    render() {
        return (
            <>
                <ProjectsNav activeKey={this.props.project.id}/>
                <Container>
                    <MLAppBar/>

                    <BreadCrumb path={this.props.project.links.app}/>
                    <Centered>
                        <h1>Find The Number</h1>
                        <p>
                            You can draw a number in the box and you'll see a boundary around that number!
                        </p><br/>
                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                    </Centered>
                    <ProjectPaginator project={this.props.project}/>
                </Container>
            </>
        )
    }
}