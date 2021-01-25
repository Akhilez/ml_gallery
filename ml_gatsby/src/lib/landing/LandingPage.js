import React from "react"
import MLLogo from "src/lib/media/ml_logo/ml_logo"
import { projectCategories } from "src/lib/globals/data"
import { projects, urls } from "../globals/data"
import { Container } from "../components/commons"
import {
  Stack,
  Button,
  Text,
  Heading,
  Divider,
  IconButton,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import ScrollMenu from "react-horizontal-scrolling-menu"
import { MetaTags } from "../components/MetaTags"
import { Project } from "./Project"

export default class LandingPage extends React.Component {
  render() {
    return (
      <Container>
        <MetaTags />
        <MLLogo />
        <this.Desc />
        {projectCategories.map(category => (
          <this.Category category={category} key={category.title} />
        ))}
      </Container>
    )
  }

  Desc() {
    return (
      <Stack alignItems="center" textAlign="center" mb="70px">
        <Text>
          Developed by{" "}
          <a href={urls.profile}>
            <b>
              <i>Akhilez</i>
            </b>
          </a>
        </Text>
        <Text mx={4} textAlign={{ base: "left", md: "center" }}>
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
        <Button
          variant="outline"
          colorScheme="brand"
          borderRadius="lg"
          as={GLink}
          to={projects.learn_line.links.app}
        >
          Take a tour
        </Button>
      </Stack>
    )
  }

  Category(props) {
    let category = props.category
    return (
      <>
        <Divider borderColor="gray.300" mb={8} />
        <Heading
          as="h2"
          textAlign="center"
          fontSize={{ base: "2xl", md: "40px" }}
          fontWeight="light"
          my={4}
        >
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
              colorScheme="red"
              variant="ghost"
              display={{ base: "none", md: "block" }}
              m={5}
            />
          }
          arrowRight={
            <IconButton
              aria-label="icon"
              icon="chevron-right"
              isRound
              size="sm"
              colorScheme="red"
              variant="ghost"
              display={{ base: "none", md: "block" }}
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
