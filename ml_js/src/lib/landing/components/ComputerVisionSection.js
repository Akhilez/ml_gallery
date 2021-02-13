import React from "react"
import {
  Box,
  Container,
  Heading,
  Text,
  Image,
  useColorModeValue,
  Button,
  Tag,
  Flex,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import SwiperCore, { Navigation, Pagination, Autoplay } from "swiper"
import { Swiper, SwiperSlide } from "swiper/react"
import {
  categoriesMap,
  projects as allProjects,
  projectStatus,
} from "../../globals/data"

import "swiper/swiper.scss"
import "swiper/components/navigation/navigation.scss"
import "swiper/components/pagination/pagination.scss"
import "swiper/components/scrollbar/scrollbar.scss"
import { IconLinks } from "./commons"

SwiperCore.use([Navigation, Pagination, Autoplay])

const ProjectSlide = ({ project }) => {
  const bg = useColorModeValue("white", "gray.700")
  return (
    <Box w="full" maxWidth="900px" mb={10} mt={4}>
      <Box
        backgroundColor={bg}
        ml={14}
        mx={12}
        boxShadow="base"
        p={4}
        borderRadius="15px"
      >
        <GLink to={project.links.app}>
          <Image
            src={require("../images/" + project.image)}
            borderRadius="10px"
          />
        </GLink>
        <Box textAlign="left" p={4}>
          <GLink to={project.links.app}>
            <Heading variant="dynamicGray" fontSize="2xl" my={2}>
              {project.title}
            </Heading>
            <Text variant="dynamicColorMode">{project.desc}</Text>
          </GLink>
          <IconLinks project={project} />
        </Box>
      </Box>
    </Box>
  )
}

export const ComputerVisionSection = () => {
  const projects = categoriesMap.vision.projects.filter(
    project => project.status !== projectStatus.toDo
  )
  return (
    <Container mt={12} py={8} align="center">
      <Heading variant="dynamicColorMode">Computer Vision</Heading>
      <Text variant="dynamicColorMode" mt={2}>
        {categoriesMap.vision.desc}
      </Text>
      <Flex my={2} justify="center">
        <Tag mx={1}>Classification</Tag>
        <Tag mx={1}>Detection</Tag>
        <Tag mx={1}>Captioning</Tag>
        <Tag mx={1}>GANs</Tag>
      </Flex>
      <Button
        colorScheme="secondary"
        size="sm"
        mt={4}
        as={GLink}
        to={allProjects.which_char.links.app}
      >
        Start here
      </Button>
      <Swiper
        spaceBetween={50}
        slidesPerView={1}
        navigation
        pagination={{ clickable: true }}
        className="vision_carousal"
        autoplay={{ delay: 3000, disableOnInteraction: true }}
      >
        {projects.map(project => (
          <SwiperSlide key={project.id}>
            <ProjectSlide project={project} />
          </SwiperSlide>
        ))}
      </Swiper>
    </Container>
  )
}
