import {Nav, Navbar} from "rsuite";
import React from "react";
import ak_logo from './media/ak_logo.svg';
import urls from '../../data/urls.json';

export default class ProfileNavBar extends React.Component {

    render() {
        return (
            <div>
                <Navbar appearance="inverse" activekey={this.props.active} className={"profile_navbar"}
                        style={{backgroundColor: "transparent"}}>
                    <Navbar.Header>
                        <a href={urls.profile.url} className="navbar-brand logo">
                            <img src={ak_logo} width={"30px"} alt={"ak_logo"} style={{paddingTop: 10}}/>
                        </a>
                    </Navbar.Header>
                    <Navbar.Body>
                        <Nav pullRight onSelect={() => {
                        }} activeKey={'activeKey'}>
                            <Nav.Item href={urls.profile.url} eventKey="profile">PROFILE</Nav.Item>
                            <Nav.Item href={urls.ml_gallery.url} eventKey="ai">ML GALLERY</Nav.Item>
                            <Nav.Item href={urls.all_projects.url} eventKey="all_projects">PROJECTS</Nav.Item>
                            <Nav.Item href={urls.resume.url} eventKey="resume">RESUME</Nav.Item>
                        </Nav>
                    </Navbar.Body>
                </Navbar>
            </div>
        );
    }
}