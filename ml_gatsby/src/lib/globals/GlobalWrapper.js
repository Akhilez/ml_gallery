import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import { Box, Flex, Text } from "@chakra-ui/core"
import "src/styles/global.sass"
import Navbar from "../components/navbar"
import { Container, Footer } from "../components/commons"
import { SideNav } from "../components/SideNav"

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

export function ProjectWrapper({ project, children, ...props }) {
  return (
    <Flex justifyContent="center" {...props}>
      <SideNav project={project} />
      <Container>{children}</Container>
    </Flex>
  )
}
