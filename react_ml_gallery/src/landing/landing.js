import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/ml_app_bar';
import MLLogo from "./ml_logo/ml_logo";
import 'cytoscape/dist/cytoscape.min';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import colorizerImage from './images/colorizer.jpg';


class LandingPage extends React.Component {
    render() {
        return (
            <Container>
                <MLAppBar/>
                <MLLogo/>
                <Project/>
                <Footer/>
            </Container>
        );
    }
}

class Project extends React.Component {
    render() {
        return (
            <div>
                <Row>
                    <Col>
                        <div className={"projectImageContainer"}>
                            <img src={colorizerImage} className={"project-image"} alt={"ColorizerImage"}/>
                        </div>
                    </Col>
                    <Col>
                        Text
                    </Col>
                </Row>
            </div>
        );
    }
}

function Footer () {
    return (
        <div><br/><hr/><br/>Footer<br/></div>
    );
}

export default LandingPage;
