import React from "react";
import {Sidenav, Nav} from 'rsuite';
import {Hidden} from "@material-ui/core";
import projects from '../../landing/data/projects';


export default class ProjectsNav extends React.Component {

    render() {
        return (
            <div style={{float: "left"}}>
                <Hidden mdDown implementation="css">
                    <Sidenav style={{backgroundColor: "transparent", marginTop: 70, float: "left", width: 200}}
                             appearance={"subtle"}>
                        <Sidenav.Header>
                            <h4 style={{color: "gray", marginLeft: 50}}>Projects</h4>
                        </Sidenav.Header>
                        <Sidenav.Body>
                            <Nav>
                                {
                                    projects.projects.map((project, index) => {
                                        return (
                                            <Nav.Item eventKey={`${project.id}`} href={project.links.app}
                                                      key={project.id}>
                                                <p style={{
                                                    margin: -5,
                                                    fontSize: 16,
                                                    color: (this.props.activeKey === project.id) ? "gray" : "#a6a6a6"
                                                }}>{index + 1}. {project.title}</p>
                                            </Nav.Item>
                                        );
                                    })
                                }
                            </Nav>
                        </Sidenav.Body>
                    </Sidenav>
                </Hidden>
            </div>
        );
    }
}
