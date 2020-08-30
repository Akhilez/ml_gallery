import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import {
  Box,
  Button,
  Flex,
  Heading,
  Image,
  Text,
  CSSReset,
} from "@chakra-ui/core"
import "src/styles/global.sass"
import Navbar from "../components/navbar"
import { Container, Footer } from "../components/commons"
import { SideNav } from "../components/SideNav"
import { BreadCrumb } from "../components/BreadCrumb"
import colabImage from "src/lib/landing/images/colab.png"
import { Link } from "gatsby"

export default class GlobalWrapper extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
  }

  render() {
    return (
      <ThemeProvider theme={theme}>
        <CSSReset />
        <Box
          fontFamily="body"
          color={theme.colors.text.default}
          fontSize="xl"
          m={2}
          className="root"
        >
          <Navbar />
          {this.props.children}
          <Footer />
        </Box>
      </ThemeProvider>
    )
  }
}

function ActionButtons({ project }) {
  return (
    <Flex>
      <Button variantColor="brand" variant="outline" size="sm" mr={2}>
        How it works
      </Button>
      <Button
        as={Link}
        to={project.links.colab}
        size="sm"
        display={project.links.colab ? "block" : "none"}
        variant="outline"
        variantColor="gray"
      >
        <Image src={colabImage} objectFit="cover" size="22px" />
      </Button>
    </Flex>
  )
}

export function ProjectWrapper({ project, children, ...props }) {
  return (
    <Container as={Flex}>
      <SideNav project={project} />
      <Box w="100%" {...props}>
        <Flex
          justifyContent="space-between"
          alignItems="center"
          direction={{ base: "column", md: "row" }}
        >
          <BreadCrumb project={project} />
          <Heading fontWeight="100">{project.title}</Heading>
          <ActionButtons project={project} />
        </Flex>
        <Text m={0} textAlign="center">
          {project.desc}
        </Text>
        {children}
      </Box>
    </Container>
  )
}
