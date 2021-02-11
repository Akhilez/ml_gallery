import React from "react"
import { Flex, Box, Text, Divider, Link } from "@chakra-ui/react"
import { projectCategories, projectStatus } from "../globals/data"
import { Link as GLink } from "gatsby"
import { SadStates } from "./SadStates"

export class ProjectPaginator extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
    this.orderedProjects = projectCategories
      .map(category => category.projects)
      .flat()
      .filter(project => project.status !== projectStatus.toDo)
    this.projectIndex = this.orderedProjects.findIndex(
      project => project.id === this.project.id
    )
    this.prevProject = this.orderedProjects[this.projectIndex - 1]
    this.nextProject = this.orderedProjects[this.projectIndex + 1]
  }

  ArrowLink = ({ href, ...props }) => (
    <Link
      as={GLink}
      to={href}
      m={2}
      p={5}
      w="50%"
      href={href}
      _hover={{ textDecoration: "none" }}
      {...props}
    >
      {props.children}
    </Link>
  )

  render() {
    return (
      <>
        <Divider mt="50px" />
        <Flex justifyContent="center" alignItems="center" w="100%">
          <SadStates
            states={[{ when: !this.prevProject, render: <Box w="50%" /> }]}
          >
            <this.ArrowLink href={this.prevProject?.links?.app} align="left">
              <Text variant="dynamicColorMode" fontSize="sm">
                Previous
              </Text>
              <Text fontSize="lg" fontWeight="bold" color="brand.300">
                {"<"} {this.prevProject?.title}
              </Text>
            </this.ArrowLink>
          </SadStates>
          <SadStates
            states={[{ when: !this.nextProject, render: <Box w="50%" /> }]}
          >
            <this.ArrowLink href={this.nextProject?.links?.app} align="right">
              <Text variant="dynamicColorMode" fontSize="sm">
                Next
              </Text>
              <Text fontSize="lg" fontWeight="bold" color="brand.300">
                {this.nextProject?.title} >
              </Text>
            </this.ArrowLink>
          </SadStates>
        </Flex>
      </>
    )
  }
}
