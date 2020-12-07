import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import { Box, CSSReset } from "@chakra-ui/core"
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
