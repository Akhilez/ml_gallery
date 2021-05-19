import React from "react"
import { Box, useColorModeValue } from "@chakra-ui/react"

export const BlockQuote = ({ children, ...props }) => {
  const bg = useColorModeValue("gray.50", "gray.700")
  return (
    <Box
      p={4}
      pl={8}
      backgroundColor={bg}
      borderRadius="lg"
      borderLeftColor="brand.200"
      borderLeftWidth={4}
      my={4}
      {...props}
    >
      {children}
    </Box>
  )
}
