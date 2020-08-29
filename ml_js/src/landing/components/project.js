import React from "react";
import Row from "react-bootstrap/Row";
import CodeIcon from "@material-ui/icons/Code";
import colabImage from "src/lib/landing/images/colab.png";
import "../landing.css";
import { Centered } from "../../commons/components/components";

export default class Project extends React.Component {
  render() {
    return (
      <div className={"ProjectContainer"}>
        <this.ProjectImage project={this.props.project} />

        <div className={"project-text-block"}>
          <h2 style={{ fontSize: 32 }}>
            <a className={"link"} href={this.props.project.links.app}>
              {this.props.project.title}
            </a>
          </h2>
          <p style={{ fontSize: 20 }}>{this.props.project.desc}</p>
          {/*this.props.project.status !== "done" && `status: ${this.props.project.status}`*/}
          {this.getIconLinks(this.props.project)}
        </div>
        {this.props.children !== null && <Row>{this.props.children}</Row>}
      </div>
    );
  }

  getIconLinks(project) {
    return (
      <div className={"row"}>
        {project.links.source && (
          <div className={"col-auto"}>
            <a className={"link"} href={project.links.app}>
              <CodeIcon />
            </a>
          </div>
        )}

        {project.links.colab && (
          <div
            className={"col-auto"}
            style={{
              backgroundImage: `url(${colabImage})`,
              backgroundPosition: "center",
              backgroundSize: "contain",
              backgroundRepeat: "no-repeat",
            }}
          >
            <a
              className={"link"}
              href={project.links.colab}
              target="_blank"
              rel="noopener noreferrer"
            >
              <div style={{ height: "28px", width: "40px" }} />
            </a>
          </div>
        )}
      </div>
    );
  }

  ProjectImage(props) {
    return (
      <Centered>
        <a href={props.project.links.app}>
          <img
            src={require("../images/" + props.project.image)}
            className={"project-image"}
            alt={props.project.title + "Image"}
          />
        </a>
      </Centered>
    );
  }
}
