import React from "react"
import {
  Button,
  Box,
  Container,
  Text,
  Heading,
  Link,
  Image,
  Wrap,
  WrapItem,
} from "@chakra-ui/react"
import { papersList } from "../../globals/data"
import { DynamicColorBox } from "../../components/dynamicColorMode"
import { HiExternalLink } from "react-icons/all"

const Paper = ({ paper }) => (
  <DynamicColorBox
    h="280px"
    m={4}
    boxShadow="base"
    borderRadius="8px"
    direction="row"
  >
    <Box w={{ base: "xs", md: "sm" }} p={4}>
      <Heading variant="dynamicGray" fontSize="2xl" mb={4}>
        {paper.title}
      </Heading>
      <Text variant="dynamicColorMode" noOfLines={5}>
        <strong>Abstract: </strong>
        {paper.abstract}
      </Text>
      <Button
        colorScheme="secondary"
        size="sm"
        mt={4}
        rightIcon={<HiExternalLink />}
        as={Link}
        href={paper.link}
        isExternal
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
        <WrapItem key={paper.title}>
          <Paper paper={paper} />
        </WrapItem>
      ))}
    </Wrap>
  </Container>
)
