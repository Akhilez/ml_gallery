import React from "react";
import { Row, Col } from "react-bootstrap";
import urls from "../../data/urls.json";
import GithubCalendar from "github-calendar";
import { Box } from "@chakra-ui/core/dist";
import { BsCode } from "react-icons/bs";
import { FaAndroid } from "react-icons/fa";
import NextLink from "next/link";

const profilePhoto = "/media/profile_photo.jpg";

export function ProfileBadge() {
  return (
    <Box className={"profileBadgeContainer"} ml={{ base: "10%", sm: 0 }}>
      <Row>
        <Col sm={"auto"}>
          <img
            src={profilePhoto}
            className={"profile_img"}
            alt={"profilePhoto"}
          />
        </Col>
        <Col sm={"auto"}>
          <div style={{ marginTop: 20 }}>
            <h1 className={"light_font"} style={{ fontSize: 58 }}>
              Akhil D. <i style={{ fontSize: 35 }}>(Akhilez)</i>
            </h1>

            <div style={{ fontSize: 28, marginTop: -10 }}>
              Deep Learning Engineer
            </div>
            <div style={{ fontSize: 22 }} className="roboto-light-ak no_href">
              <a href="mailto: akhilez.ai@gmail.com">
                {" "}
                akhilez.ai<span style={{ color: "#8d9599" }}>@gmail.com</span>
              </a>
            </div>
            <Social />
            <ResumeButton />
          </div>
        </Col>
      </Row>
    </Box>
  );
}

export function ResumeButton() {
  return (
    <div>
      <NextLink href={urls.resume.url}>
        <a className="btn btn-outline-secondary resume-button">RESUME</a>
      </NextLink>
    </div>
  );
}

export function Social(props) {
  return (
    <div className="social">
      <ul>
        <li>
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://www.linkedin.com/in/akhilez/"
          >
            <i className="fa fa-linkedin" />
          </a>
        </li>
        <li>
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://github.com/Akhilez"
          >
            <i className="fa fa-github" />
          </a>
        </li>
      </ul>
    </div>
  );
}

export class MyGithubCalendar extends React.Component {
  render() {
    return (
      <div
        className="calendar"
        style={{ marginTop: 20, marginBottom: 50, width: "100%" }}
      >
        Loading the data just for you.
      </div>
    );
  }

  componentDidMount() {
    const _ = GithubCalendar(".calendar", "Akhilez", { responsive: true });
  }
}

export function ProjectBox(props) {
  let project = props.data;
  return (
    <div className="no_href">
      <div className="row project_box" key={project.title}>
        <div className="col-md-5">
          <a target="_blank" rel="noopener noreferrer" href={project.links.app}>
            <img
              className="project_image"
              src={`/media/profile/projects/${project.image}.png`}
              alt={project.image}
              width="400px"
            />
          </a>
        </div>
        <div className="col-md-7">
          <h4 className="project_title">
            <a
              target="_blank"
              rel="noopener noreferrer"
              href={project.links.app}
            >
              {project.title}
            </a>
          </h4>
          <div className={"projectLinkText"}>
            <a
              href={project.links.app}
              style={{ color: "#919c9e", fontWeight: 400 }}
              target={"_blank"}
              rel="noopener noreferrer"
            >
              {project.links.app}
            </a>
          </div>
          <p className="project_description">{project.desc}</p>

          <div className="row">
            {project.tags.map((tag) => (
              <div className="col-auto chip_tag" key={tag}>
                {tag}
              </div>
            ))}
          </div>

          <div className="row">
            <div className="col-auto project_date">{project.date}</div>
            {project.links.code != null && (
              <div
                className="col-auto view_source_button"
                data-toggle="tooltip"
                title="View source code"
              >
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href={project.links.code}
                >
                  <Box as={BsCode} mt="2px" />
                </a>
              </div>
            )}
            {project.links.android && (
              <div
                className="col-auto view_source_button"
                data-toggle="tooltip"
                title="Android App"
              >
                <a
                  target="_blank"
                  rel="noopener noreferrer"
                  href={project.links.android}
                >
                  <Box as={FaAndroid} mt="2px" />
                </a>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
