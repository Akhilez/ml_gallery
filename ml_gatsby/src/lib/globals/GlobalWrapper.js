import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import { Box } from "@chakra-ui/core"
import "src/styles/global.css"

export default class GlobalWrapper extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
  }
  render() {
    return (
      <React.StrictMode>
        <ThemeProvider theme={theme}>
          <Box fontFamily="body" color={theme.colors.text.default}>
            {this.props.children}
          </Box>
        </ThemeProvider>
      </React.StrictMode>
    )
  }
}
