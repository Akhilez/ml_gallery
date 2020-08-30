import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import { Box, Button, Flex, IconButton, Image, Text } from "@chakra-ui/core"
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
      <React.StrictMode>
        <ThemeProvider theme={theme}>
          <Box
            fontFamily="body"
            color={theme.colors.text.default}
            fontSize="xl"
            className="root"
          >
            <Navbar />
            {this.props.children}
            <Footer />
          </Box>
        </ThemeProvider>
      </React.StrictMode>
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
    <Flex justifyContent="center" {...props}>
      <SideNav project={project} />
      <Container>
        <Flex justifyContent="space-between" alignItems="center">
          <BreadCrumb project={project} />
          <Text>{project.title}</Text>
          <ActionButtons project={project} />
        </Flex>
        <Text>{project.description}</Text>
        {children}
      </Container>
    </Flex>
  )
}
