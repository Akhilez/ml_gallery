import React from "react"
import {
  Accordion,
  AccordionButton,
  AccordionIcon,
  AccordionItem,
  AccordionPanel,
  Box,
  Text,
  Link,
  Collapse,
  IconButton,
  useDisclosure,
  useBreakpointValue,
} from "@chakra-ui/react"
import { projectCategories, projectStatus } from "../globals/data"
import { Link as GLink } from "gatsby"
import { BsChevronCompactLeft, BsChevronCompactRight } from "react-icons/all"
import { DynamicColorBox } from "./dynamicColorMode"

export const SideNav = ({ project, ...props }) => (
  <Box w="250px" color="gray.400" mr={2} fontSize="sm" {...props}>
    <Text variant="dynamicColorMode" fontSize="lg" mb={4} ml={4}>
      Projects
    </Text>
    <Accordion allowMultiple>
      {projectCategories.map(category => (
        <AccordionItem key={category.title}>
          <AccordionButton border={0} backgroundColor="transparent">
            <Box flex="1" textAlign="left">
              <Text variant="dynamicColorMode">{category.title}</Text>
            </Box>
            <AccordionIcon />
          </AccordionButton>
          <AccordionPanel pb={4}>
            {category.projects
              .filter(project => project.status !== projectStatus.toDo)
              .map(project => (
                <Box key={project.title}>
                  <Link
                    as={GLink}
                    to={project.links.app}
                    color="gray.500"
                    _hover={{ textDecoration: "none" }}
                  >
                    {project.title}
                  </Link>
                </Box>
              ))}
          </AccordionPanel>
        </AccordionItem>
      ))}
    </Accordion>
  </Box>
)

export const CollapsibleSideNav = ({ project, ...props }) => {
  const dynamicOpen = useBreakpointValue({ base: false, xl: true })
  const { onToggle, isOpen } = useDisclosure({ isOpen: dynamicOpen })
  return (
    <DynamicColorBox
      direction="row"
      position="absolute"
      top="56px"
      left="0"
      boxShadow="base"
      borderRightRadius="xl"
      align="center"
      p={2}
      {...props}
    >
      <Collapse in={isOpen} unmountOnExit animateOpacity>
        <SideNav project={project} pt={2} pb={8} />
      </Collapse>

      <IconButton
        icon={isOpen ? <BsChevronCompactLeft /> : <BsChevronCompactRight />}
        variant="ghost"
        minW="20px"
        size="sm"
        colorScheme="brand"
        onClick={onToggle}
      />
    </DynamicColorBox>
  )
}
