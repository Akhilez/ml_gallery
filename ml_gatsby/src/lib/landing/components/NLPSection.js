import React from "react"
import {
  Box,
  Image,
  Flex,
  Divider,
  Button,
  Text,
  Heading,
  SimpleGrid,
  Wrap,
  WrapItem,
  Tag,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { Centered, Container } from "../../components/commons"
import { categoriesMap, projects } from "../../globals/data"
import { DynamicColorBox } from "../../components/dynamicColorMode"
import { IconLinks } from "./commons"

const Project = ({ project }) => (
  <DynamicColorBox
    m={4}
    p={2}
    boxShadow="xl"
    borderRadius="10px"
    width="220px"
    minH="312px"
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
      <Text
        variant="dynamicColorMode"
        lineHeight={1.25}
        fontSize="sm"
        mt={2}
        noOfLines={3}
      >
        {project.desc}
      </Text>
      <IconLinks project={project} />
    </Box>
  </DynamicColorBox>
)

export const NLPSection = () => (
  <Container my={8}>
    <Box ml={4} mb={4}>
      <Heading mb={2} variant="dynamicColorMode">
        Natural Language Processing
      </Heading>
      <Text variant="dynamicColorMode">{categoriesMap.nlp.desc}</Text>
      <Flex my={2}>
        <Tag mr={1}>RNNs</Tag>
        <Tag mx={1}>Embeddings</Tag>
        <Tag mx={1}>Attention</Tag>
        <Tag mx={1}>Classification</Tag>
      </Flex>
      <Button
        colorScheme="secondary"
        size="sm"
        mt={2}
        as={GLink}
        to={projects.next_char.links.app}
      >
        Get started
      </Button>
    </Box>
    <Wrap>
      {categoriesMap.nlp.projects.map(project => (
        <WrapItem key={project.id}>
          <Project project={project} />
        </WrapItem>
      ))}
    </Wrap>
  </Container>
)
