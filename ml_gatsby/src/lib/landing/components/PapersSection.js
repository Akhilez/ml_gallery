import React from "react"
import {
  Button,
  Box,
  Text,
  Divider,
  Heading,
  Flex,
  Image,
  Wrap,
  WrapItem,
} from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import { papersList } from "../../globals/data"
import { DynamicColorBox } from "../../components/dynamicColorMode"
import { HiExternalLink } from "react-icons/all"

const Paper = ({ paper }) => (
  <DynamicColorBox
    h="280px"
    m={4}
    boxShadow="xl"
    borderLeftRadius="8px"
    borderRightRadius={{ base: "8px", md: 0 }}
    direction="row"
  >
    <Box w={{ base: "xs", md: "lg" }} p={4}>
      <Heading variant="dynamicGray" fontSize="2xl" mb={4}>
        {paper.title}
      </Heading>
      <Text variant="dynamicColorMode" noOfLines={5}>
        {paper.abstract}
      </Text>
      <Button
        colorScheme="secondary"
        size="sm"
        mt={4}
        rightIcon={<HiExternalLink />}
      >
        Read
      </Button>
    </Box>
    <Box>
      <Image
        display={{ base: "none", sm: "block" }}
        src={require("../images/" + paper.image)}
        alt={paper.title + "Image"}
        h="100%"
        borderRightRadius="8px"
        w="150px"
      />
    </Box>
  </DynamicColorBox>
)

export const PapersSection = () => (
  <Container>
    <Heading variant="dynamicColorMode" ml={4} mb={4}>
      Unpublished Papers
    </Heading>
    <Wrap>
      {papersList.map(paper => (
        <WrapItem key={paper.id}>
          <Paper paper={paper} />
        </WrapItem>
      ))}
    </Wrap>
  </Container>
)
