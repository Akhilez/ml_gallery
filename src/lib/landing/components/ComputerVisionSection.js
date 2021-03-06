import React from "react"
import {
  Box,
  Container,
  Heading,
  Text,
  Image,
  IconButton,
  useColorModeValue,
  Button,
  Tag,
  Flex,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
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
import {
  ButtonBack,
  ButtonNext,
  CarouselProvider,
  CarouselContext,
  Slide,
  Slider,
} from "pure-react-carousel"
import "pure-react-carousel/dist/react-carousel.es.css"
import { AiFillCaretLeft, AiFillCaretRight } from "react-icons/all"
import { Dots, HCarouselControls } from "./common"

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
            <Text>{project.desc}</Text>
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
      <Text mt={2}>{categoriesMap.vision.desc}</Text>
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
      <Box w="full" maxWidth="900px">
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
                <ProjectSlide project={project} />
              </Slide>
            ))}
          </Slider>
          <HCarouselControls projects={projects} />
        </CarouselProvider>
      </Box>
    </Container>
  )
}
