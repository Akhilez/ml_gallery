import React from "react"
import { Text, Box, Stack, Flex } from "@chakra-ui/core"
import { Container } from "src/lib/components/commons"
import { projects } from "src/lib/globals/data"

export default class ComingSoon extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.learn_line
  }
  render() {
    return (
      <Flex justifyContent="center">
        <Box w="250px" bg="teal.500" display={{ base: "none", xl: "block" }}>
          <Text>Hello this is a box</Text>
        </Box>
        <Container bg="red.500">
          <Text>Hello there</Text>
        </Container>
      </Flex>
    )
  }
}
