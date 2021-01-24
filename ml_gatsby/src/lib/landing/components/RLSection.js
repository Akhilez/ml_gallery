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

SwiperCore.use([Pagination])

const RLTag = ({ name }) => (
  <WrapItem>
    <Tag>{name}</Tag>
  </WrapItem>
)

const LeftSection = () => (
  <Flex w={{ base: "100%", md: "50%" }} justify="flex-end" mr={8}>
    <Box w={{ base: "full", md: "md", lg: "lg" }} p={4}>
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
  <Box>
    <Image
      src={require("../images/" + project.image)}
      alt={project.title + "Image"}
      height="150px"
    />
    <Heading variant="dynamicGray" fontSize="lg">
      {project.title}
    </Heading>
    <Text>{project.desc}</Text>
  </Box>
)

const RightSection = () => {
  const bg = useColorModeValue("brand.500", "brand.800")
  return (
    <Flex
      borderLeftRadius="40px"
      backgroundColor={bg}
      w={{ base: "100%", md: "50%" }}
      h="500px"
    >
      <Swiper
        spaceBetween={50}
        slidesPerView={1}
        pagination={{ clickable: true }}
        className="vision_carousal"
        direction="vertical"
      >
        {categoriesMap.reinforce.projects.map(project => (
          <SwiperSlide key={project.id}>
            <RLProject project={project} />
          </SwiperSlide>
        ))}
      </Swiper>
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
