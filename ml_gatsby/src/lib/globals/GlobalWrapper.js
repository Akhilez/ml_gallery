import React from "react"
import ThemeProvider from "@chakra-ui/core/dist/ThemeProvider"
import theme from "./theme"
import { Box } from "@chakra-ui/core"

export default class GlobalWrapper extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
  }
  render() {
    return (
      <React.StrictMode>
        <ThemeProvider theme={theme}>
          <Box>{this.props.children}</Box>
        </ThemeProvider>
      </React.StrictMode>
    )
  }
}
