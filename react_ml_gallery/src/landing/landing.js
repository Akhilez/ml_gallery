import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/ml_app_bar';
import MLLogo from "./ml_logo/ml_logo";
import 'cytoscape/dist/cytoscape.min';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import colorizerImage from './images/colorizer.jpg';
import projectsData from './data/projects';


class LandingPage extends React.Component {
    render() {
        return (
            <div  className={"page"}>
            <Container>
                <MLAppBar/>
                <MLLogo/>
                <Project project={projectsData.projects[0]} image={colorizerImage}/>
                <Footer/>
            </Container>
            </div>
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
                            <a href={this.props.project.links.source}>
                            <img src={this.props.image} className={"project-image"}
                                 alt={this.props.project.title + "Image"}/></a>
                        </div>
                    </Col>
                    <Col>
                        <div className={"project-text-block"}>
                        <h2>{this.props.project.title}</h2>
                        <p>{this.props.project.desc}</p>
                        </div>
                    </Col>
                </Row>
                {this.props.children !== null &&
                <Row>{this.props.children}</Row>
                }
            </div>
        );
    }
}

function Footer() {
    return (
        <div><br/>
            <hr/>
            <br/>Footer<br/></div>
    );
}

export default LandingPage;
