import React from "react";
import { Container } from "react-bootstrap";
import ProfileNavBar from "../navbar";
import projectsData from "../data/projects.json";
import { ProjectBox } from "../profile_components";

export default class AllProjectsPage extends React.Component {
  render() {
    return (
      <div className={"profile_root"}>
        <Container>
          <ProfileNavBar active={"all_projects"} />
          <h1 style={{ marginTop: 70 }}>Projects</h1>
          <this.Projects />
        </Container>
      </div>
    );
  }

  Projects(props) {
    let deployed_projects = projectsData.projects.filter(
      (project) => project.status === "deployed"
    );
    return (
      <div>
        {deployed_projects.map((project) => (
          <ProjectBox data={project} key={project.title} />
        ))}
      </div>
    );
  }
}
