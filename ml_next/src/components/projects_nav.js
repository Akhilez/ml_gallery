import React from "react";
import { Sidenav, Nav, Dropdown } from "rsuite";
import { Hidden } from "@material-ui/core";
import { projects } from "../pages/gallery/project";

export default class ProjectsNav extends React.Component {
  render() {
    let openId = null;
    for (let i = 0; i < projects.categories.length; i++)
      for (let project of projects.categories[i].projects)
        if (project.id === this.props.activeKey) {
          openId = i;
          break;
        }
    return (
      <div style={{ float: "left" }}>
        <Hidden mdDown implementation="css">
          <Sidenav
            style={{
              backgroundColor: "transparent",
              marginTop: 70,
              float: "left",
              width: 300,
              marginLeft: -30,
              color: "gray",
            }}
            appearance={"subtle"}
            defaultOpenKeys={[`${openId}`]}
          >
            <Sidenav.Header>
              <h4 style={{ color: "gray", marginLeft: 54 }}>Projects</h4>
            </Sidenav.Header>
            <Sidenav.Body>
              <Nav>
                {projects.categories.map((category, indexOuter) => (
                  <Dropdown
                    eventKey={`${indexOuter}`}
                    title={category.title}
                    key={indexOuter}
                    style={{ color: openId === indexOuter ? "black" : "gray" }}
                  >
                    {category.projects.map((project, index) => (
                      <Dropdown.Item
                        eventKey={`${project.id}`}
                        href={project.links.app}
                        key={project.id}
                      >
                        <p
                          style={{
                            margin: -5,
                            marginLeft: 10,
                            fontSize: 16,
                            color:
                              this.props.activeKey === project.id
                                ? "black"
                                : "gray",
                          }}
                        >
                          {index + 1}. {project.title}
                        </p>
                      </Dropdown.Item>
                    ))}
                  </Dropdown>
                ))}
              </Nav>
            </Sidenav.Body>
          </Sidenav>
        </Hidden>
      </div>
    );
  }
}
