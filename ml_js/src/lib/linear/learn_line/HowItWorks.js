import { Box, Divider, Heading, Text } from "@chakra-ui/react"
import React from "react"

export function HowItWorks() {
  return (
    <Box>
      <Divider my={6} />
      <Heading fontSize="2xl" id="how_it_works">
        How It Works
      </Heading>
      <Text>
        This article is written keeping in mind that you are a beginner. So I
        will dive into the basics first and then get into the meat of the
        subject. Well, actually I'll write this part later. Come back after ...
        maybe a month.
      </Text>
    </Box>
  )
}
