import React from "react"
import { Box, Flex, Heading, Image, Link, Stack, Text } from "@chakra-ui/core"
import { Link as GLink } from "gatsby"
import { MdCode } from "react-icons/md"
import colabImage from "./images/colab.png"

export class Project extends React.Component {
  render() {
    return (
      <Stack
        width={{ base: "220px", md: "300px" }}
        bg="white"
        pt={2}
        px={3}
        borderRadius="15px"
        mx={2}
        minH={{ base: "sm", md: "450px" }}
        className="ProjectContainer"
      >
        <this.ProjectImage project={this.props.project} />
        <Box className="project-text-block">
          <Heading as="h2" fontSize={{ base: "lg", md: "xl" }} mb={2}>
            <Link as={GLink} to={this.props.project.links.app}>
              {this.props.project.title}
            </Link>
          </Heading>
          <Text fontSize={14} mb={2}>
            {this.props.project.desc}
          </Text>
          {this.getIconLinks(this.props.project)}
        </Box>

        {this?.props?.children}
      </Stack>
    )
  }

  getIconLinks(project) {
    return (
      <Flex alignItems="center">
        {project.links.source && (
          <a className={"link"} href={project.links.app}>
            <Box as={MdCode} fontSize="2xl" mt={1} />
          </a>
        )}

        {project.links.colab && (
          <Link
            href={project.links.colab}
            target="_blank"
            rel="noopener noreferrer"
          >
            <Box
              backgroundImage={`url(${colabImage})`}
              backgroundPosition="center"
              backgroundSize="contain"
              backgroundRepeat="no-repeat"
              h="28px"
              w="40px"
            />
          </Link>
        )}
      </Flex>
    )
  }

  ProjectImage(props) {
    return (
      <Stack alignItems="center">
        <GLink to={props.project.links.app}>
          <Image
            src={require("./images/" + props.project.image)}
            alt={props.project.title + "Image"}
            width="300px"
            height={{ base: "150px", md: "220px" }}
            className={"project-image"}
            minW={{ base: "200px", md: "280px" }}
          />
        </GLink>
      </Stack>
    )
  }
}
