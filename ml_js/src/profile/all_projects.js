import React from "react";
import {Container} from "react-bootstrap";
import ProfileNavBar from "./navbar";

export default class AllProjectsPage extends React.Component {
    render() {
        return (
            <div className={"profile_root"}>
                <Container>
                    <ProfileNavBar active={"all_projects"}/>
                    <h1>Projects</h1>

                </Container>
            </div>
        );
    }
}