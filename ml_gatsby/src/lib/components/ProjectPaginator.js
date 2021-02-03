import React from "react"
import { Flex, Box, Text, Divider } from "@chakra-ui/react"
import { SolidLink } from "./commons"
import { projectCategories, projectStatus } from "../globals/data"

export class ProjectPaginator extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
    this.orderedProjects = projectCategories
      .map(category => category.projects)
      .flat()
      .filter(project => project.status !== projectStatus.toDo)
    console.log(this.orderedProjects)
    this.projectIndex = this.orderedProjects.findIndex(
      project => project.id === this.project.id
    )
    this.prevProject = this.orderedProjects[this.projectIndex - 1]
    this.nextProject = this.orderedProjects[this.projectIndex + 1]
  }
  render() {
    return (
      <>
        <Divider mt="50px" />
        <Flex justifyContent="center" alignItems="center">
          {this.prevProject && (
            <SolidLink href={this.prevProject?.links?.app} m={2} p={5} w="50%">
              <Box textAlign="right">
                <Text fontSize="sm">Previous</Text>
                <Text>{this.prevProject?.title}</Text>
              </Box>
            </SolidLink>
          )}
          {this.nextProject && (
            <SolidLink href={this.nextProject?.links?.app} m={2} p={5} w="50%">
              <Box fontSize="sm">Next</Box>
              <Text>{this.nextProject?.title}</Text>
            </SolidLink>
          )}
        </Flex>
      </>
    )
  }
}
