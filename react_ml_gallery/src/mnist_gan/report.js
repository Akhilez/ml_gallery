import MLAppBar from "../commons/ml_app_bar";
import React from "react";
import Container from "react-bootstrap/Container";
import BreadCrumb from "../commons/components/breadcrumb";
import '../landing/landing.css';


export default class MnistGanReport extends React.Component {
    constructor(props) {
        super(props);
        this.path = '/mnist_gan/report';
    }
    render() {
        return (
            <div className={"page"}>
            <Container>
                <MLAppBar/>
                <BreadCrumb path={this.path}/>
                REPORT
            </Container>
            </div>
        );
    }
}