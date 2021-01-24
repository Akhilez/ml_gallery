import React from "react"
import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  Image,
  useColorModeValue,
  Link,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { categoriesMap, projects, urls } from "../../globals/data"
import { Container } from "../../components/commons"

const LeftSection = () => (
  <Flex w={{ base: "100%", md: "50%" }} justify="flex-end">
    <Box w="md">
      <Heading>RL</Heading>
      <Text>Hellooo</Text>
      <Button>Get started</Button>
    </Box>
  </Flex>
)

const RightSection = () => {
  const bg = useColorModeValue("brand.500", "brand.800")
  return (
    <Flex
      borderLeftRadius="40px"
      backgroundColor={bg}
      w={{ base: "100%", md: "50%" }}
    >
      <Box h="400px" />
    </Flex>
  )
}

export const RLSection = () => {
  return (
    <Flex justify="center" alignItems="center" mt="100px" mb="150px">
      <LeftSection />
      <RightSection />
    </Flex>
  )
}
