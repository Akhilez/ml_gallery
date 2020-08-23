import React from "react"
import Navbar from "src/lib/components/navbar"
import MLLogo from "src/lib/media/ml_logo/ml_logo"
import { projectCategories } from "src/lib/globals/data"
import { urls } from "../globals/data"
import colabImage from "src/lib/landing/images/colab.png"
import { Container, Footer } from "../components/commons"
import { Stack, Button, SimpleGrid } from "@chakra-ui/core"
import { Link as GLink } from "gatsby"
import { MdCode } from "react-icons/md"

export default class LandingPage extends React.Component {
  render() {
    return (
      <Container>
        <Navbar />
        <MLLogo />
        <this.Desc />
        {projectCategories.map(category => (
          <this.Category category={category} key={category.title} />
        ))}
        <Footer />
      </Container>
    )
  }

  Desc() {
    return (
      <Stack alignItems="center">
        <div style={{ fontSize: 22, marginBottom: 70 }}>
          <p>
            Developed by{" "}
            <a href={urls.profile.url}>
              <b>
                <i>Akhilez</i>
              </b>
            </a>
          </p>
          <p>
            <b>Machine Learning Gallery</b> is a master project of few of my
            experiments with Neural Networks. It is designed in a way to help a
            beginner understand the concepts with visualizations. You can train
            and run the networks live and see the results for yourself. Every
            project here is followed by an explanation on how it works.
            <br />
            <br />
            Begin with a tour starting from the most basic Neural Network and
            build your way up.
          </p>
          <Button variant="outline" as={GLink} to={"/learn_line"}>
            Take a tour
          </Button>
        </div>
      </Stack>
    )
  }

  Category(props) {
    let category = props.category
    return (
      <div>
        <hr />
        <br />
        <h3 className={"ProjectCategoryTitle"}>{category.title}</h3>
        <SimpleGrid display="inline">
          {category.projects.map(project => (
            <Project project={project} key={project.id} />
          ))}
        </SimpleGrid>
        <br />
      </div>
    )
  }
}

class Project extends React.Component {
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
        {this?.props?.children}
      </div>
    )
  }

  getIconLinks(project) {
    return (
      <div className={"row"}>
        {project.links.source && (
          <div className={"col-auto"}>
            <a className={"link"} href={project.links.app}>
              <MdCode />
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
    )
  }

  ProjectImage(props) {
    return (
      <Stack alignItems="center">
        <a href={props.project.links.app}>
          <img
            src={require("./images/" + props.project.image)}
            className={"project-image"}
            alt={props.project.title + "Image"}
          />
        </a>
      </Stack>
    )
  }
}
