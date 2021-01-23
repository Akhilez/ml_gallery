import React from "react"
import { Box, Heading, Text, Image, useColorModeValue } from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import SwiperCore, { Navigation, Pagination } from "swiper"
import { Swiper, SwiperSlide } from "swiper/react"
import { categoriesMap } from "../../globals/data"

import "swiper/swiper.scss"
import "swiper/components/navigation/navigation.scss"
import "swiper/components/pagination/pagination.scss"
import "swiper/components/scrollbar/scrollbar.scss"

SwiperCore.use([Navigation, Pagination])

const ProjectSlide = ({ project }) => {
  const bg = useColorModeValue("white", "gray.700")
  return (
    <Box w="full" my={10}>
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
    <Container my={12} py={8}>
      <Centered>
        <Heading variant="dynamicColorMode">Computer Vision</Heading>
        <Swiper
          spaceBetween={50}
          slidesPerView={1}
          navigation
          pagination={{ clickable: true }}
          className="vision_carousal"
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
