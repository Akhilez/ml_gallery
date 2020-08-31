import React from "react"
import { Flex, Box, Text } from "@chakra-ui/core"
import { orderedProjects } from "../globals/data"
import { SolidLink } from "./commons"

export class ProjectPaginator extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
    this.projectIndex = orderedProjects.findIndex(
      project => project.id === this.project.id
    )
    this.prevProject = orderedProjects[this.projectIndex - 1]
    this.nextProject = orderedProjects[this.projectIndex + 1]
  }
  render() {
    console.log(this.prevProject)
    console.log(this.nextProject)
    return (
      <Flex justifyContent="center" alignItems="center">
        {this.prevProject && (
          <SolidLink href={this.prevProject?.links?.app} m={2} p={5}>
            <Box>
              <Text fontSize="sm">Previous</Text>
              <Text>{this.prevProject?.title}</Text>
            </Box>
          </SolidLink>
        )}
        {this.nextProject && (
          <SolidLink href={this.nextProject?.links?.app} m={2} p={5}>
            <Box fontSize="sm">Next</Box>
            <Text>{this.nextProject?.title}</Text>
          </SolidLink>
        )}
      </Flex>
    )
  }
}
