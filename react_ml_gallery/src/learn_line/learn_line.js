import React from "react";
import Container from "react-bootstrap/Container";
import MLAppBar from "../commons/ml_app_bar";
import {Centered} from "../commons/components/components";
import '../landing/landing.css';


export default class LearnLinePage extends React.Component {
    render () {
        return (
            <Container>
                <MLAppBar />
                <Centered>
                    <h1>Learn A Line</h1>
                    <p>Predict the m and c values of the straight line (y = mx + c) equation.</p>
                </Centered>
            </Container>
        );
    }
}

