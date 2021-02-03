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

export const BrandFlex = ({ children, ...props }) => {
  const bg = useColorModeValue("brand.500", "brand.800")
  return (
    <Flex backgroundColor={bg} {...props}>
      {children}
    </Flex>
  )
}
