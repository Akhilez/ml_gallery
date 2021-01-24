import React from "react"
import { useColorModeValue, Flex } from "@chakra-ui/react"

export const DynamicColorBox = ({ children, ...props }) => {
  const bg = useColorModeValue("white", "gray.700")
  return (
    <Flex backgroundColor={bg} direction="column" {...props}>
      {children}
    </Flex>
  )
}
