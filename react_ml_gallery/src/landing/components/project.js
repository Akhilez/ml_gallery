import React from "react";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";

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
                            <h2 style={{fontSize: 42}}>{this.props.project.title}</h2>
                            <p style={{fontSize: 20}}>{this.props.project.desc}</p>
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
                <a href={props.project.links.app}>
                    <img src={require('../images/' + props.project.image)} className={"project-image"}
                         alt={props.project.title + "Image"}/></a>
            </div>
        );
    }
}


export default Project;