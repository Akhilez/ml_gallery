import React from "react"
import { Box, Flex } from "@chakra-ui/react"

export const BasicsSection = () => {
  return (
    <Flex direction={{ base: "column", md: "row-reverse" }}>
      <Box backgroundColor="brand.500" h="sm" w={{ base: "100%", md: "50%" }}>
        hello
      </Box>
      <Box backgroundColor="secondary.500" h="sm" w="50%">
        hello
      </Box>
    </Flex>
  )
}
