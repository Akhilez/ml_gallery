import React from "react"
import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  Image,
  useColorModeValue,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { projects } from "../../globals/data"

const RightSection = () => {
  return (
    <Flex
      direction="column"
      w={{ base: "100%", md: "50%" }}
      p={{ base: 8, md: "100px" }}
      justify="center"
    >
      <Box>
        <Heading>The Basics</Heading>
        <Text my={2}>
          Start your Deep Learning journey with simple problems
        </Text>
        <Button my={2} size="sm" colorScheme="secondary">
          Start with basics
        </Button>
      </Box>
    </Flex>
  )
}

const BasicProject = ({ project }) => {
  const bg = useColorModeValue("backgroundColor", "gray.800")
  return (
    <Flex backgroundColor={bg} m={6} borderRadius="14px" boxShadow="xl">
      <Box p={2}>
        <GLink to={project.links.app}>
          <Image
            src={require("../images/" + project.image)}
            alt={project.title + "Image"}
            width="120px"
            height="100px"
            className={"project-image"}
            borderRadius="8px"
            minW="50px"
          />
        </GLink>
      </Box>
      <Box p={2}>
        <Heading variant="dynamicGray" fontSize="lg" my={2}>
          {project.title}
        </Heading>
        <Text variant="dynamicColorMode" fontSize="sm" lineHeight="1">
          {project.desc}
        </Text>
      </Box>
    </Flex>
  )
}

const LeftSection = () => {
  const bg = useColorModeValue("brand.500", "brand.800")
  return (
    <Flex
      borderRightRadius="40px"
      backgroundColor={bg}
      w={{ base: "100%", md: "50%" }}
      justify="flex-end"
    >
      <Box w={{ base: "sm", sm: "lg" }} py={6} pr={8}>
        <BasicProject project={projects.learn_line} />
        <BasicProject project={projects.linear_classifier} />
        <BasicProject project={projects.learn_curve} />
        <BasicProject project={projects.deep_iris} />
      </Box>
    </Flex>
  )
}

export const BasicsSection = () => {
  return (
    <Flex direction={{ base: "column", md: "row-reverse" }}>
      <RightSection />
      <LeftSection />
    </Flex>
  )
}
