import React from "react"
import {
  Accordion,
  AccordionHeader,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Text,
  Link,
} from "@chakra-ui/core"
import { projectCategories } from "../globals/data"
import { Link as GLink } from "gatsby"

export class SideNav extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
  }

  render() {
    return (
      <Box
        w="250px"
        display={{ base: "none", xl: "block" }}
        color="gray.400"
        mr={4}
        mt="50px"
        fontSize="sm"
      >
        <Text fontSize="lg" mb={4} ml={4}>
          Projects
        </Text>
        <Accordion allowMultiple>
          {projectCategories.map(category => (
            <AccordionItem key={category.title}>
              <AccordionHeader border={0} backgroundColor="transparent">
                <Box flex="1" textAlign="left">
                  <Text>{category.title}</Text>
                </Box>
                <AccordionIcon />
              </AccordionHeader>
              <AccordionPanel pb={4}>
                {category.projects.map(project => (
                  <Box key={project.title}>
                    <Link as={GLink} to={project.links.app} color="gray.400">
                      {project.title}
                    </Link>
                  </Box>
                ))}
              </AccordionPanel>
            </AccordionItem>
          ))}
        </Accordion>
      </Box>
    )
  }
}
