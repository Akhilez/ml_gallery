import React from "react"
import {
  Box,
  Button,
  Flex,
  Heading,
  Text,
  Image,
  Tag,
  useColorModeValue,
  Wrap,
  WrapItem,
  IconButton,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { categoriesMap, projects, projectStatus } from "../../globals/data"
import {
  ButtonBack,
  ButtonNext,
  CarouselProvider,
  DotGroup,
  Slide,
  Slider,
} from "pure-react-carousel"
import { AiFillCaretLeft, AiFillCaretRight } from "react-icons/all"

const RLTag = ({ name }) => (
  <WrapItem>
    <Tag>{name}</Tag>
  </WrapItem>
)

const LeftSection = () => (
  <Flex w={{ base: "100%", md: "50%" }} justify="flex-end" mr={8}>
    <Box
      w={{ base: "full", md: "md", lg: "lg", xl: "xl", "2xl": "2xl" }}
      p={4}
      pl={16}
    >
      <Heading variant="dynamicColorMode">Reinforcement Learning</Heading>
      <Text mt={2} mb={4}>
        Learn how models are trained to play games from basic to
        state-of-the-art methods
      </Text>
      <Wrap>
        <RLTag name="Policy Gradients" />
        <RLTag name="Deep Q" />
        <RLTag name="MCTS" />
        <RLTag name="AlphaZero" />
        <RLTag name="MuZero" />
      </Wrap>
      <Button
        mt={4}
        colorScheme="secondary"
        size="sm"
        as={GLink}
        to={projects.alpha_nine.links.app}
      >
        Get started
      </Button>
    </Box>
  </Flex>
)

const RLProject = ({ project }) => (
  <Box width={{ base: "sm", md: "lg" }}>
    <Box py={16}>
      <GLink to={project.links.app}>
        <Image
          src={require("../images/" + project.image)}
          alt={project.title + "Image"}
          maxWidth="90%"
          maxHeight="250px"
          borderRadius="8px"
        />
        <Heading color="white" fontSize="2xl" mt={4} mb={2}>
          {project.title}
        </Heading>
        <Text color="gray.100">{project.desc}</Text>
      </GLink>
    </Box>
  </Box>
)

const RightSection = () => {
  const projects = categoriesMap.reinforce.projects.filter(
    project => project.status !== projectStatus.toDo
  )
  const bg = useColorModeValue(
    "linear(to-br, brand.500, red.500)",
    "linear(to-br, brand.700, red.700)"
  )

  return (
    <Flex
      direction="column"
      w={{ base: "100%", md: "50%" }}
      h="500px"
      bgGradient={bg}
      borderLeftRadius="40px"
    >
      <Box h="500px" align="center">
        <CarouselProvider
          visibleSlides={1}
          totalSlides={projects.length}
          naturalSlideWidth={400}
          isIntrinsicHeight
          isPlaying
          interval={3000}
        >
          <Slider>
            {projects.map((project, idx) => (
              <Slide index={idx} key={project.id}>
                <RLProject project={project} />
              </Slide>
            ))}
          </Slider>
          <CarouselControls projects={projects} />
        </CarouselProvider>
      </Box>
    </Flex>
  )
}

export const RLSection = () => {
  return (
    <Flex
      justify="center"
      alignItems="center"
      my="100px"
      direction={{ base: "column", md: "row" }}
    >
      <LeftSection />
      <RightSection />
    </Flex>
  )
}

const Dots = ({ projects, ...props }) => (
  <DotGroup
    renderDots={({ currentSlide, carouselStore }) => (
      <Flex justify="center" {...props}>
        {projects.map((project, index) => (
          <StyledDot
            onClick={() => carouselStore.setStoreState({ currentSlide: index })}
            active={currentSlide === index}
          />
        ))}
      </Flex>
    )}
  />
)

const StyledDot = props => {
  const bg_active = useColorModeValue("secondary.500", "secondary.200")

  return (
    <Box
      w={3}
      h={3}
      m={1}
      borderRadius="full"
      backgroundColor={props.active ? bg_active : "blackAlpha.300"}
      {...props}
    />
  )
}

const CarouselControls = ({ projects }) => (
  <Flex align="center" justify="center">
    <ButtonBack>
      <IconButton
        icon={<AiFillCaretLeft />}
        size="sm"
        variant="ghost"
        isRound
        mx={2}
        colorScheme="secondary"
      />
    </ButtonBack>
    <Dots projects={projects} />
    <ButtonNext>
      <IconButton
        icon={<AiFillCaretRight />}
        size="sm"
        variant="ghost"
        isRound
        mx={2}
        colorScheme="secondary"
      />
    </ButtonNext>
  </Flex>
)
