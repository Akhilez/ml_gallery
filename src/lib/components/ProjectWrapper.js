import {
  Button,
  Flex,
  Heading,
  Image,
  Link as CLink,
  Container,
  Wrap,
} from "@chakra-ui/react"
import colabImage from "../landing/images/colab.png"
import { BreadCrumb } from "./BreadCrumb"
import { CollapsibleSideNav } from "./SideNav"
import { ProjectPaginator } from "./ProjectPaginator"
import React from "react"
import { MetaTags } from "./MetaTags"

function ActionButtons({ project }) {
  return (
    <Flex my={2}>
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

export function ProjectWrapper({ project, children, ...props }) {
  return (
    <>
      <Container minH="90vh" {...props}>
        <MetaTags title={`${project.title} | ML Gallery`} />
        <Wrap justify="space-between">
          <BreadCrumb project={project} ml={{ base: 6, md: 0 }} />
          <ActionButtons project={project} />
        </Wrap>
        <Heading variant="dynamicColorMode" mt={4}>
          {project.title}
        </Heading>
        {children}
        <ProjectPaginator project={project} />
      </Container>
      <CollapsibleSideNav project={project} />
    </>
  )
}
