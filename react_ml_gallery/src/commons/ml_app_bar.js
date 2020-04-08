import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import ak_logo from '../commons/images/AK_logo.svg';
import '../landing/landing.css';


class MLAppBar extends React.Component {
    render() {
        return (
            <Navbar bg="transparent" variant="light">
                <Nav className="mr-auto">
                    <Navbar.Brand href="http://akhilez.com/"><img src={ak_logo} alt={"AK Logo"} height={"40px"} /></Navbar.Brand>
                </Nav>
                <Nav.Link href="/" className={"nav-link"}><div>HOME</div></Nav.Link>
            </Navbar>
        );
    }
}

export default MLAppBar;