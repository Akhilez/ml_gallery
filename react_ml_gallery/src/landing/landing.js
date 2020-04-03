import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/ml_app_bar';
import logo from './images/ml_logo.png';


class LandingPage extends React.Component {
    render() {
        return (
            <Container>
                <MLAppBar/>
                <MLLogo/>
                <Project/>
            </Container>
        );
    }
}

class MLLogo extends React.Component {
    render() {
        let imgStyle = {
            marginTop: "100px",
            marginBottom: "100px",
        };
        return (
            <div align="center">
                <img alt="ml_logo" src={logo} height={"300px"} width={"400px"} style={imgStyle}/>
            </div>
        );
    }
}

class Project extends React.Component {
    render() {
        return (
            <div>
                Project
            </div>
        );
    }
}

export default LandingPage;
