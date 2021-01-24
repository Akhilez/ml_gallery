import React from "react"
import {
  Box,
  Heading,
  Text,
  Image,
  useColorModeValue,
  Button,
} from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import SwiperCore, { Navigation, Pagination, Autoplay } from "swiper"
import { Swiper, SwiperSlide } from "swiper/react"
import { categoriesMap } from "../../globals/data"

import "swiper/swiper.scss"
import "swiper/components/navigation/navigation.scss"
import "swiper/components/pagination/pagination.scss"
import "swiper/components/scrollbar/scrollbar.scss"

SwiperCore.use([Navigation, Pagination, Autoplay])

const ProjectSlide = ({ project }) => {
  const bg = useColorModeValue("white", "gray.700")
  return (
    <Box w="full" maxWidth="900px" mb={10} mt={4}>
      <Box
        backgroundColor={bg}
        ml={14}
        mx={12}
        boxShadow="xl"
        p={4}
        borderRadius="15px"
      >
        <Image
          src={require("../images/" + project.image)}
          borderRadius="10px"
        />
        <Box textAlign="left" p={4}>
          <Heading variant="dynamicGray" fontSize="2xl" my={2}>
            {project.title}
          </Heading>
          <Text variant="dynamicColorMode">{project.desc}</Text>
        </Box>
      </Box>
    </Box>
  )
}

export const ComputerVisionSection = () => {
  return (
    <Container mt={12} py={8}>
      <Centered>
        <Heading variant="dynamicColorMode">Computer Vision</Heading>
        <Text variant="dynamicColorMode" mt={2}>
          {categoriesMap.vision.desc}
        </Text>
        <Button colorScheme="secondary" size="sm" mt={4}>
          Start here
        </Button>
        <Swiper
          spaceBetween={50}
          slidesPerView={1}
          navigation
          pagination={{ clickable: true }}
          className="vision_carousal"
          autoplay={{ delay: 2500, disableOnInteraction: true }}
        >
          {categoriesMap.vision.projects.map(project => (
            <SwiperSlide key={project.id}>
              <ProjectSlide project={project} />
            </SwiperSlide>
          ))}
        </Swiper>
      </Centered>
    </Container>
  )
}
