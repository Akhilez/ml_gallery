import React from "react"
import { Box, ChakraProvider } from "@chakra-ui/react"
import theme from "./theme"
import "src/styles/global.sass"
import Navbar from "../components/navbar"
import { Footer } from "../components/commons"
import "typeface-roboto-condensed"

export default class GlobalWrapper extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
  }

  render() {
    return (
      <ChakraProvider resetCSS theme={theme}>
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
      </ChakraProvider>
    )
  }
}
