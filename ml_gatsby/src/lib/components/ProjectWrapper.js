import {
  Button,
  Flex,
  Heading,
  Image,
  Link as CLink,
  Text,
} from "@chakra-ui/react"
import colabImage from "../landing/images/colab.png"
import { Container } from "./commons"
import { BreadCrumb } from "./BreadCrumb"
import { ProjectsNavigatorFrame } from "./SideNav"
import { ProjectPaginator } from "./ProjectPaginator"
import React from "react"
import { MetaTags } from "./MetaTags"

function ActionButtons({ project }) {
  return (
    <Flex>
      <Button
        as={CLink}
        href="#how_it_works"
        colorScheme="brand"
        variant="outline"
        boxSize="sm"
        mr={2}
        fontWeight="light"
      >
        How it works
      </Button>
      <Button
        as={CLink}
        href={project.links.colab}
        boxSize="sm"
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

export function ProjectWrapper({ project, children, ...props }) {
  return (
    <Container>
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
