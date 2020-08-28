import React from "react"
import Navbar from "src/lib/components/navbar"
import MLLogo from "src/lib/media/ml_logo/ml_logo"
import { projectCategories } from "src/lib/globals/data"
import { urls } from "../globals/data"
import colabImage from "src/lib/landing/images/colab.png"
import { Container, Footer } from "../components/commons"
import {
  Stack,
  Button,
  Text,
  Image,
  Heading,
  Divider,
  Link,
  IconButton,
} from "@chakra-ui/core"
import { Link as GLink } from "gatsby"
import { MdCode } from "react-icons/md"
import ScrollMenu from "react-horizontal-scrolling-menu"

class Project extends React.Component {
  render() {
    return (
      <Stack width="sm" bg="white" p={4} borderRadius="15px" mx={2}>
        <this.ProjectImage project={this.props.project} />
        <Heading as="h2" fontSize="lg" mb={2}>
          <Link as={GLink} to={this.props.project.links.app} fontSize="28px">
            {this.props.project.title}
          </Link>
        </Heading>
        <Text>{this.props.project.desc}</Text>
        {this.getIconLinks(this.props.project)}

        {this?.props?.children}
      </Stack>
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
        <GLink to={props.project.links.app}>
          <Image
            src={require("./images/" + props.project.image)}
            alt={props.project.title + "Image"}
            width="sm"
          />
        </GLink>
      </Stack>
    )
  }
}

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
      <Stack alignItems="center" textAlign="center" mb="70px">
        <Text>
          Developed by{" "}
          <a href={urls.profile.url}>
            <b>
              <i>Akhilez</i>
            </b>
          </a>
        </Text>
        <Text>
          <b>Machine Learning Gallery</b> is a master project of few of my
          experiments with Neural Networks. It is designed in a way to help a
          beginner understand the concepts with visualizations. You can train
          and run the networks live and see the results for yourself. Every
          project here is followed by an explanation on how it works.
          <br />
          <br />
          Begin with a tour starting from the most basic Neural Network and
          build your way up.
        </Text>
        <Button variant="outline" as={GLink} to={"/learn_line"}>
          Take a tour
        </Button>
      </Stack>
    )
  }

  Category(props) {
    let category = props.category
    return (
      <>
        <Divider borderColor="gray.300" />
        <br />
        <Heading as="h2" textAlign="center" fontSize="40px" fontWeight="light">
          {category.title}
        </Heading>
        <ScrollMenu
          data={category.projects.map((project, index) => (
            <Project project={project} key={index} />
          ))}
          arrowLeft={
            <IconButton
              aria-label="icon"
              icon="chevron-left"
              isRound
              size="sm"
              variantColor="red"
              variant="ghost"
              m={5}
            />
          }
          arrowRight={
            <IconButton
              aria-label="icon"
              icon="chevron-right"
              isRound
              size="sm"
              variantColor="red"
              variant="ghost"
              m={5}
            />
          }
          hideSingleArrow={true}
          hideArrows={true}
          inertiaScrolling={true}
          useButtonRole={false}
        />
        <br />
      </>
    )
  }
}
