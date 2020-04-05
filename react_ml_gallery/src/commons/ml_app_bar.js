import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import Form from "react-bootstrap/Form";
import Button from "react-bootstrap/Button";
import ak_logo from '../commons/images/AK_logo.svg';


class MLAppBar extends React.Component {
    render() {
        return (
            <Navbar bg="transparent" variant="light">
                <Navbar.Brand href="http://akhilez.com/"><img src={ak_logo} alt={"AK Logo"} height={"40px"} /></Navbar.Brand>
                <Nav className="mr-auto">
                    <Nav.Link href="#home">Home</Nav.Link>
                    <Nav.Link href="#features">Features</Nav.Link>
                    <Nav.Link href="#pricing">Pricing</Nav.Link>
                </Nav>
                <Form inline>
                    <Button variant="outline-primary">Search</Button>
                </Form>
            </Navbar>
        );
    }
}

export default MLAppBar;