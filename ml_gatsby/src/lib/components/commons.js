import React from "react"
import { Divider, Text, useTheme } from "@chakra-ui/core"
import Box from "@chakra-ui/core/dist/Box"

export const Container = ({ children, ...props }) => {
  const theme = useTheme()

  console.log(theme.breakpoints)
  console.log("slice")
  console.log(theme.breakpoints.slice(1))

  return (
    <Box {...props} mx="auto" maxW={{ base: "full", xl: "60em" }} w="100%">
      {children}
    </Box>
  )
}

export function Footer() {
  return (
    <Container>
      <Divider />
      <Text>ML Gallery</Text>
    </Container>
  )
}
