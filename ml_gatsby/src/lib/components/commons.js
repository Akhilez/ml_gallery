import React from "react"
import { Text, useTheme } from "@chakra-ui/core"
import Box from "@chakra-ui/core/dist/Box"

export const Container = ({ children, ...props }) => {
  const theme = useTheme()

  return (
    <Box
      {...props}
      mx="auto"
      w={["full", "full", ...theme.breakpoints.slice(1)]}
    >
      {children}
    </Box>
  )
}

export function Footer() {
  return (
    <Box>
      <hr />
      <Text>ML Gallery</Text>
    </Box>
  )
}
