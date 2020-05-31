import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import ak_logo from '../images/AK_logo.svg';
import '../../landing/landing.css';


class MLAppBar extends React.Component {
    render() {
        return (
            <Navbar bg="transparent" variant="light">
                <Nav className="mr-auto">
                    <Navbar.Brand href="http://akhilez.com/"><img src={ak_logo} alt={"AK Logo"} height={"40px"} /></Navbar.Brand>
                </Nav>
                <Nav.Link href="/" className={"nav-link"}><div>HOME</div></Nav.Link>
                <Nav.Link href="/profile" className={"nav-link"}><div>PROFILE</div></Nav.Link>
                <Nav.Link href="https://github.com/Akhilez/ml_gallery" className={"nav-link"} target={"_blank"}><div>REPO</div></Nav.Link>
            </Navbar>
        );
    }
}

export default MLAppBar;