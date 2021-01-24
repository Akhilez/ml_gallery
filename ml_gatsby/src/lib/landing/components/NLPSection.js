import React from "react"
import {
  Box,
  Image,
  Flex,
  Divider,
  Text,
  Heading,
  SimpleGrid,
  Wrap,
  WrapItem,
} from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import { categoriesMap } from "../../globals/data"
import { DynamicColorBox } from "../../components/dynamicColorMode"

const Project = ({ project }) => (
  <DynamicColorBox
    m={4}
    p={2}
    boxShadow="xl"
    borderRadius="10px"
    width="220px"
    h="350px"
  >
    <Image
      src={require("../images/" + project.image)}
      alt={project.title + "Image"}
      height="150px"
      borderRadius="8px"
    />
    <Box textAlign="left" p={2}>
      <Heading variant="dynamicGray" fontSize="lg" my={2}>
        {project.title}
      </Heading>
      <Text variant="dynamicColorMode" lineHeight={1.25} fontSize="sm" mt={2}>
        {project.desc}
      </Text>
    </Box>
  </DynamicColorBox>
)

export const NLPSection = () => (
  <Container my={8}>
    <Heading mb={4} ml={4}>
      Natural Language Processing
    </Heading>
    <Wrap>
      {categoriesMap.nlp.projects.map(project => (
        <WrapItem key={project.id}>
          <Project project={project} />
        </WrapItem>
      ))}
    </Wrap>
  </Container>
)
