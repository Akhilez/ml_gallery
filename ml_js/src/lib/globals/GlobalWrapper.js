import React from "react"
import { Box, ChakraProvider } from "@chakra-ui/react"
import theme from "./theme"
import "src/styles/global.sass"
import { StaticNavbar } from "../components/navbar"
import { Footer } from "../components/commons"
import "typeface-roboto-condensed"

const GlobalWrapper = ({ children }) => (
  <ChakraProvider resetCSS theme={theme}>
    <StaticNavbar />
    {children}
    <Footer />
  </ChakraProvider>
)

export default GlobalWrapper
