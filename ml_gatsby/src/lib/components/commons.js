import React from "react"
import { Divider, Text, useTheme } from "@chakra-ui/core"
import Box from "@chakra-ui/core/dist/Box"

export const Container = ({ children, ...props }) => {
  const theme = useTheme()

  return (
    <Box
      {...props}
      mx="auto"
      maxW={["full", "full", ...theme.breakpoints.slice(1)]}
      w="100%"
    >
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
