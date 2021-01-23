import React from "react"
import { Box, Stack } from "@chakra-ui/react"
import { MetaTags } from "../components/MetaTags"
import { Centered, Container } from "../components/commons"

export const LandingPageV2 = () => (
  <Box>
    <MetaTags />
    <Container>
      <IntroSection />
    </Container>
  </Box>
)

const IntroSection = () => (
  <Stack
    direction={{ base: "column", md: "row-reverse" }}
    justify="center"
    align="center"
  >
    <Box w="md" h="md" backgroundColor="brand.500">
      Hi
    </Box>
    <Box w="md" h="md" backgroundColor="secondary.500">
      Hello
    </Box>
  </Stack>
)
