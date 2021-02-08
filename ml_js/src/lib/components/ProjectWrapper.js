import {
  Button,
  Flex,
  Heading,
  Image,
  Link as CLink,
  Box,
  Text,
  IconButton,
  Container,
  SimpleGrid,
  Wrap,
  useDisclosure,
  DrawerOverlay,
  DrawerContent,
  DrawerCloseButton,
  Drawer,
} from "@chakra-ui/react"
import colabImage from "../landing/images/colab.png"
import { BreadCrumb } from "./BreadCrumb"
import { CollapsibleSideNav, ProjectsNavigatorFrame, SideNav } from "./SideNav"
import { ProjectPaginator } from "./ProjectPaginator"
import React from "react"
import { MetaTags } from "./MetaTags"

function ActionButtons({ project }) {
  return (
    <Flex my={2}>
      <Button
        as={CLink}
        href="#how_it_works"
        colorScheme="brand"
        variant="outline"
        size="sm"
        mr={2}
        fontWeight="light"
      >
        How it works
      </Button>
      <Button
        as={CLink}
        href={project.links.colab}
        size="sm"
        display={project.links.colab ? "block" : "none"}
        variant="outline"
        colorScheme="gray"
        isExternal
      >
        <Image src={colabImage} objectFit="cover" boxSize="23px" mt="3px" />
      </Button>
    </Flex>
  )
}

export function ProjectWrapperOld({ project, children, ...props }) {
  return (
    <Container minH="90vh">
      <MetaTags title={`${project.title} | ML Gallery`} />
      <Flex
        justifyContent="space-between"
        alignItems="center"
        direction={{ base: "column", md: "row" }}
      >
        <BreadCrumb project={project} />
        <Heading fontWeight="100" ml={{ base: 0, xl: "150px" }}>
          {project.title}
        </Heading>
        <ActionButtons project={project} />
      </Flex>
      <ProjectsNavigatorFrame {...props}>
        <Text m={0} textAlign="center">
          {project.desc}
        </Text>
        {children}
        <ProjectPaginator project={project} />
      </ProjectsNavigatorFrame>
    </Container>
  )
}

export function ProjectWrapper({ project, children, ...props }) {
  return (
    <>
      <Container minH="90vh" {...props}>
        <MetaTags title={`${project.title} | ML Gallery`} />
        <Wrap justify="space-between">
          <BreadCrumb project={project} ml={{ base: 6, md: 0 }} />
          <ActionButtons project={project} />
        </Wrap>
        <Heading mt={4}>{project.title}</Heading>
        {children}
      </Container>
      <CollapsibleSideNav project={project} />
    </>
  )
}
