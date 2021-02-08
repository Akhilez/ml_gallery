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
  Flex,
  Drawer,
  DrawerOverlay,
  Collapse,
  Slide,
  DrawerContent,
  DrawerCloseButton,
  IconButton,
  useDisclosure,
  useBreakpointValue,
} from "@chakra-ui/react"
import { projectCategories, projectStatus } from "../globals/data"
import { Link as GLink } from "gatsby"
import {
  FiMenu,
  BsChevronCompactLeft,
  BsChevronCompactRight,
} from "react-icons/all"

export class SideNav extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
  }

  render() {
    return (
      <Box w="250px" color="gray.400" mr={4} fontSize="sm" {...this.props}>
        <Text fontSize="lg" mb={4} ml={4}>
          Projects
        </Text>
        <Accordion allowMultiple>
          {projectCategories.map(category => (
            <AccordionItem key={category.title}>
              <AccordionButton border={0} backgroundColor="transparent">
                <Box flex="1" textAlign="left">
                  <Text>{category.title}</Text>
                </Box>
                <AccordionIcon />
              </AccordionButton>
              <AccordionPanel pb={4}>
                {category.projects
                  .filter(project => project.status !== projectStatus.toDo)
                  .map(project => (
                    <Box key={project.title}>
                      <Link as={GLink} to={project.links.app} color="gray.400">
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
  }
}

export const CollapsibleSideNav = ({ project, ...props }) => {
  const dynamicOpen = useBreakpointValue({ base: false, xl: true })
  console.log(dynamicOpen)
  const { onToggle, isOpen } = useDisclosure({ isOpen: dynamicOpen })
  return (
    <Flex
      position="absolute"
      top="50px"
      left="0"
      backgroundColor="white"
      boxShadow="base"
      roundedRight="xl"
      align="center"
      p={2}
      {...props}
    >
      <Collapse in={isOpen} unmountOnExit animateOpacity>
        <SideNav project={project} pb={8} />
      </Collapse>

      <IconButton
        icon={isOpen ? <BsChevronCompactLeft /> : <BsChevronCompactRight />}
        variant="ghost"
        minW="20px"
        size="sm"
        colorScheme="brand"
        onClick={onToggle}
      />
    </Flex>
  )
}

export class ProjectsNavigatorFrame extends React.Component {
  constructor(props) {
    super(props)
    this.props = props
    this.project = props.project
    this.state = {
      isDrawerOpen: false,
    }
  }
  render() {
    return (
      <Flex>
        <this.OpenDrawerButton />
        <SideNav
          project={this.project}
          display={{ base: "none", xl: "block" }}
        />
        <Box w="100%" {...this.props}>
          {this.props.children}
        </Box>
      </Flex>
    )
  }
  OpenDrawerButton = () => {
    return (
      <Box
        position="absolute"
        top="100px"
        left={5}
        display={{ base: "block", xl: "none" }}
        onClick={() => this.setState({ isDrawerOpen: true })}
      >
        <FiMenu />
        <Drawer
          isOpen={this.state.isDrawerOpen}
          placement="left"
          onClose={() => this.setState({ isDrawerOpen: false })}
        >
          <DrawerOverlay />
          <DrawerContent width="270px">
            <DrawerCloseButton />
            <SideNav project={this.project} />
          </DrawerContent>
        </Drawer>
      </Box>
    )
  }
}
