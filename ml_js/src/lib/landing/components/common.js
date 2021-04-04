import { ButtonBack, ButtonNext, DotGroup } from "pure-react-carousel"
import { Box, Flex, IconButton, useColorModeValue } from "@chakra-ui/react"
import React from "react"
import { AiFillCaretLeft, AiFillCaretRight } from "react-icons/all"

export const Dots = ({ projects, ...props }) => (
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
  const bg_inactive = useColorModeValue("gray.200", "gray.700")
  const bg_active = useColorModeValue("secondary.500", "secondary.200")

  return (
    <Box
      w={3}
      h={3}
      m={1}
      borderRadius="full"
      backgroundColor={props.active ? bg_active : bg_inactive}
      {...props}
    />
  )
}

export const HCarouselControls = ({ projects }) => (
  <Flex justify="center" align="center">
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
