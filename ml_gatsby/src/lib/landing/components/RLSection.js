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
  Link,
  Wrap,
  WrapItem,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { categoriesMap, projects, urls } from "../../globals/data"
import { Container } from "../../components/commons"
import { Swiper, SwiperSlide } from "swiper/react"
import SwiperCore, { Navigation, Pagination } from "swiper"

import "swiper/swiper.scss"
import "swiper/components/navigation/navigation.scss"
import "swiper/components/pagination/pagination.scss"
import "swiper/components/scrollbar/scrollbar.scss"
import { BrandFlex } from "../../components/dynamicColorMode"

SwiperCore.use([Pagination])

const RLTag = ({ name }) => (
  <WrapItem>
    <Tag>{name}</Tag>
  </WrapItem>
)

const LeftSection = () => (
  <Flex w={{ base: "100%", md: "50%" }} justify="flex-end" mr={8}>
    <Box w={{ base: "full", md: "md", "2xl": "3xl" }} p={4} pl={16}>
      <Heading variant="dynamicColorMode">Reinforcement Learning</Heading>
      <Text variant="dynamicColorMode" mt={2} mb={4}>
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
      <Button mt={4} colorScheme="secondary" size="sm">
        Get started
      </Button>
    </Box>
  </Flex>
)

const RLProject = ({ project }) => (
  <Box width={{ base: "sm", md: "lg" }} py={16} pl={16}>
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
    <Text color="white">{project.desc}</Text>
  </Box>
)

const RightSection = () => {
  return (
    <BrandFlex
      direction="column"
      borderLeftRadius="40px"
      w={{ base: "100%", md: "50%" }}
      h="500px"
    >
      <Swiper
        spaceBetween={50}
        slidesPerView={1}
        pagination={{ clickable: true }}
        className="vision_carousal"
        direction="vertical"
        style={{ marginLeft: 0 }}
      >
        {categoriesMap.reinforce.projects.map(project => (
          <SwiperSlide key={project.id}>
            <RLProject project={project} />
          </SwiperSlide>
        ))}
      </Swiper>
    </BrandFlex>
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
