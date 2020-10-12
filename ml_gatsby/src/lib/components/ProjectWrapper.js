import {
  Box,
  Button,
  Flex,
  Heading,
  Image,
  Link as CLink,
  Text,
} from "@chakra-ui/core"
import colabImage from "../landing/images/colab.png"
import { Container } from "./commons"
import { BreadCrumb } from "./BreadCrumb"
import { SideNav } from "./SideNav"
import { ProjectPaginator } from "./ProjectPaginator"
import React from "react"
import { MetaTags } from "./MetaTags"

function ActionButtons({ project }) {
  return (
    <Flex>
      <Button
        as={CLink}
        href="#how_it_works"
        variantColor="brand"
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
        variantColor="gray"
        isExternal
      >
        <Image src={colabImage} objectFit="cover" size="23px" mt="3px" />
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
      <Flex>
        <SideNav project={project} />
        <Box w="100%" {...props}>
          <Text m={0} textAlign="center">
            {project.desc}
          </Text>
          {children}
          <ProjectPaginator project={project} />
        </Box>
      </Flex>
    </Container>
  )
}
