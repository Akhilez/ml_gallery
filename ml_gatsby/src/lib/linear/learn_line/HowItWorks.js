import { Box, Divider, Heading, Text } from "@chakra-ui/core"
import React from "react"

export function HowItWorks() {
  return (
    <Box>
      <Divider />
      <Heading fontSize="2xl" id="how_it_works">
        How It Works
      </Heading>
      <Text>
        This article is written keeping in mind that you are a beginner.
      </Text>
    </Box>
  )
}
