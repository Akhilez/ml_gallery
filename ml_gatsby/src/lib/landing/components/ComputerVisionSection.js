import React from "react"
import { Box, Heading, Text } from "@chakra-ui/react"
import { Centered, Container } from "../../components/commons"
import SwiperCore, { Navigation, Pagination, Scrollbar, A11y } from "swiper"
import { Swiper, SwiperSlide } from "swiper/react"
import { categoriesMap } from "../../globals/data"

import "swiper/swiper.scss"
import "swiper/components/navigation/navigation.scss"
import "swiper/components/pagination/pagination.scss"
import "swiper/components/scrollbar/scrollbar.scss"

SwiperCore.use([Navigation, Pagination])

const ProjectSlide = ({ project }) => {
  return (
    <Box backgroundColor="red.300" w="full" h="lg">
      {project.title}
    </Box>
  )
}

export const ComputerVisionSection = () => {
  return (
    <Container my={12} py={8} px={2}>
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
