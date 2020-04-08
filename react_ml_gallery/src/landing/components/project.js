import React from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import projImage from '../images/colorizer.jpg';

class Project extends React.Component {
    render() {
        return (
            <div className={"ProjectContainer"}>
                <Row>
                    <Col>
                        <this.ProjectImage project={this.props.project} />
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

    ProjectImage(props){
        return (
            <div className={"projectImageContainer"}>
                <a href={props.project.links.source}>
                    <img src={require('../images/' + props.project.image)} className={"project-image"}
                         alt={props.project.title + "Image"}/></a>
            </div>
        );
    }
}


export default Project;