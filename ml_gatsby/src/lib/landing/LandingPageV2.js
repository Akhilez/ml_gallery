import React from "react"
import {
  Box,
  Button,
  Image,
  Text,
  Link,
  Heading,
  Flex,
  useColorModeValue,
} from "@chakra-ui/react"
import { MetaTags } from "../components/MetaTags"
import { Container } from "../components/commons"
import mlg_logo from "../media/ml_logo/ml_logo.png"
import { urls } from "../globals/data"
import { BasicsSection } from "./components/BasicsSection"
import { ComputerVisionSection } from "./components/ComputerVisionSection"
import { NLPSection } from "./components/NLPSection"
import { RLSection } from "./components/RLSection"
import { PapersSection } from "./components/PapersSection"
import { UpcomingSection } from "./components/UpcomingSection"

export const LandingPageV2 = () => (
  <Box>
    <MetaTags />
    <IntroSection />
    <BasicsSection />
    <ComputerVisionSection />
    <NLPSection />
    <RLSection />
    <PapersSection />
    <UpcomingSection />
  </Box>
)

const IntroSection = () => {
  const headingColor = useColorModeValue("brand.500", "brand.400")
  const line1Color = useColorModeValue("gray.500", "gray.300")
  const line2Color = useColorModeValue("gray.600", "gray.200")
  return (
    <Container>
      <Flex
        direction={{ base: "column", md: "row-reverse" }}
        justify="center"
        alignItems="center"
        mt="100px"
        mb="150px"
      >
        <Box
          w={{ base: "sm", md: "md" }}
          align="center"
          justify="center"
          alignItems="center"
        >
          <Image src={mlg_logo} />
        </Box>
        <Box w="100px" h="50px" />
        <Box w={{ base: "sm", md: "md" }}>
          <Heading fontWeight="500" color={headingColor}>
            Machine Learning Gallery
          </Heading>
          <Text fontSize="sm" fontWeight="bold" color={line1Color} mt={2}>
            Developed by <Link href={urls.profile}>Akhilez</Link>
          </Text>
          <Text mt={2} color={line2Color}>
            This is a master project of some experiments with Neural Networks.
            Every project here is runnable, visualized and explained clearly.
          </Text>
          <Button mt={4} colorScheme="secondary" size="sm">
            Take a tour
          </Button>
        </Box>
      </Flex>
    </Container>
  )
}
