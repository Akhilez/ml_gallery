import React from "react"
import { ChakraProvider } from "@chakra-ui/react"
import theme from "./theme"
import "src/styles/global.sass"
import { StaticNavbar } from "../components/navbar"
import { Footer } from "../components/commons"
import "typeface-roboto-condensed"
import { QueryClient, QueryClientProvider } from "react-query"

const queryClient = new QueryClient({
  defaultOptions: { queries: { refetchOnWindowFocus: false, retry: false } },
})

const GlobalWrapper = ({ children }) => (
  <QueryClientProvider client={queryClient}>
    <ChakraProvider theme={theme}>
      <StaticNavbar />
      {children}
      <Footer />
    </ChakraProvider>
  </QueryClientProvider>
)

export default GlobalWrapper
