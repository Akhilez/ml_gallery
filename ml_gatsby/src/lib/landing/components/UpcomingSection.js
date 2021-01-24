import React from "react"
import { Container } from "../../components/commons"
import {
  Divider,
  Heading,
  Box,
  UnorderedList,
  ListItem,
} from "@chakra-ui/react"
import { projectCategories, projectStatus } from "../../globals/data"
import { Text } from "recharts"

const ToDoSection = ({ category }) => (
  <Box>
    <Heading variant="dynamicGray" mt={6} fontWeight="bold">
      {category.title}
    </Heading>
    <Divider width="md" my={2} />
    <UnorderedList spacing={1}>
      {category.toDoProjects.map(project => (
        <ListItem>
          <Text variant="dynamicColorMode">{project.title}</Text>
        </ListItem>
      ))}
    </UnorderedList>
  </Box>
)

export const UpcomingSection = () => {
  const toDoProjects = projectCategories
    .map(category => ({
      title: category.title,
      toDoProjects: category.projects.filter(
        project => project?.status === projectStatus.toDo
      ),
    }))
    .filter(category => category.toDoProjects?.length > 0)
  return (
    <Container my={16} p={4}>
      <Heading variant="dynamicColorMode">More to come</Heading>
      {toDoProjects.map(category => (
        <ToDoSection category={category} />
      ))}
    </Container>
  )
}
