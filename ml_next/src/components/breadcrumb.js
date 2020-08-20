import React from "react";
import routes from "../pages/routes.json";
import { projects } from "../pages/gallery/project";
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import { MdNavigateNext } from "react-icons/md/index";

export default class BreadCrumb extends React.Component {
  render() {
    let links = this.getLinks(this.props.path);
    return (
      <div className={"breadcrumbContainer"}>
        <Row>
          {links.map((link, index) => {
            return (
              <Col xs="auto" key={index}>
                <Row>
                  <Col
                    xs="auto"
                    style={{ padding: "5px 10px" }}
                    className={"breadcrumbLink"}
                  >
                    <a className={"link"} href={link.link}>
                      {link.title}
                    </a>
                  </Col>
                  <Col xs="auto" style={{ padding: 0 }}>
                    {index !== links.length - 1 && (
                      <MdNavigateNext className={"breadcrumbIcon"} />
                    )}
                  </Col>
                </Row>
              </Col>
            );
          })}
        </Row>
      </div>
    );
  }

  getLinks(path) {
    /*
        1. Check if the any project has this path.
        2. Check if the path exists in routes.
         */

    let links = [];
    projects.categories.forEach((category) => {
      category.projects.forEach((project) => {
        if (project.links.app === path) {
          links = [
            { title: "Home", link: "/" },
            { title: project.title, link: project.links.app },
          ];
        }
      });
    });

    if (links.length > 0) return links;

    let pages = path.split("/");
    let tree = routes.tree;
    for (let i = 0; i < pages.length; i++) {
      let link = "/" + pages[i];
      tree = tree.children[link];
      links.push({
        title: tree.title,
        link: tree.link,
      });
    }
    return links;
  }
}
