import React from "react"
import { useColorModeValue, Box } from "@chakra-ui/react"

export const DynamicColorBox = ({ children, ...props }) => {
  const bg = useColorModeValue("white", "gray.700")
  return (
    <Box backgroundColor={bg} {...props}>
      {children}
    </Box>
  )
}
