import React from "react"
import {
  Box,
  Button,
  Container,
  Image,
  Text,
  Link,
  Heading,
  Avatar,
  Flex,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { MetaTags } from "../components/MetaTags"
import mlg_logo from "../media/ml_logo/ml_logo.png"
import avatarPhoto from "./images/smiling_akhil.jpeg"
import { projects, urls } from "../globals/data"
import { BasicsSection } from "./components/BasicsSection"
import { ComputerVisionSection } from "./components/ComputerVisionSection"
import { NLPSection } from "./components/NLPSection"
import { RLSection } from "./components/RLSection"
import { PapersSection } from "./components/PapersSection"
import { UpcomingSection } from "./components/UpcomingSection"

export const LandingPage = () => (
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

const IntroSection = () => (
  <Container>
    <Flex
      direction={{ base: "column", md: "row-reverse" }}
      justify="center"
      alignItems="center"
      mt="100px"
      mb="150px"
    >
      <Box w={{ base: "sm", md: "md" }} align="center">
        <Image src={mlg_logo} />
      </Box>
      <Box w="100px" h="50px" />
      <Box w={{ base: "sm", md: "md" }}>
        <Heading fontWeight="500" variant="dynamicColorMode">
          Machine Learning Gallery
        </Heading>
        <Flex my={2} fontSize="sm">
          <Avatar name="Akhil Devarashetti" src={avatarPhoto} mr={4} />
          <Box>
            <Text fontWeight="bold" mt={2}>
              Developed by Akhil Devarashetti
            </Text>
            <Text>
              Deep Learning Engineer | <Link href={urls.profile}>akhil.ai</Link>
            </Text>
          </Box>
        </Flex>
        <Text mt={2}>
          This is a master project of some experiments with Neural Networks.
          Every project here is runnable, visualized and explained clearly.
        </Text>
        <Button
          mt={4}
          colorScheme="secondary"
          size="sm"
          as={GLink}
          to={projects.learn_line.links.app}
        >
          Take a tour
        </Button>
      </Box>
    </Flex>
  </Container>
)
