import {Nav, Navbar} from "rsuite";
import React from "react";
import './css/profile_style.css';
import ak_logo from './media/ak_logo.svg';

export default class ProfileNavBar extends React.Component {

    render() {
        return (
            <div>
                <Navbar appearance="inverse" activeKey={this.props.active} className={"profile_navbar"}
                        style={{backgroundColor: "transparent"}}>
                    <Navbar.Header>
                        <a href="#" className="navbar-brand logo">
                            <img src={ak_logo} width={"30px"} alt={"ak_logo"} style={{paddingTop: 10}}/>
                        </a>
                    </Navbar.Header>
                    <Navbar.Body>
                        <Nav pullRight onSelect={() => {
                        }} activeKey={'activeKey'}>
                            <Nav.Item href="/profile" eventKey="profile">PROFILE</Nav.Item>
                            <Nav.Item href="/" eventKey="ai">MACHINE LEARNING</Nav.Item>
                            <Nav.Item href="/profile/all_projects" eventKey="projects">PROJECTS</Nav.Item>
                            <Nav.Item href="/resume" eventKey="resume">RESUME</Nav.Item>
                        </Nav>
                    </Navbar.Body>
                </Navbar>
            </div>
        );
    }
}