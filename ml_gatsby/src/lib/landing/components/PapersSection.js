import React from "react"
import { Box, Divider, Heading, Flex } from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import { papersList } from "../../globals/data"

const Paper = ({ paper }) => <Box>{paper.title}</Box>

export const PapersSection = () => (
  <Container>
    <Divider w="2xl" />
    <Heading variant="dynamicColorMode">Unpublished Papers</Heading>
    <Flex>
      {papersList.map(paper => (
        <Paper paper={paper} />
      ))}
    </Flex>
  </Container>
)
