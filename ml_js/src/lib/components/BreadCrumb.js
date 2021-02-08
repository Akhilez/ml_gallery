import React from "react"
import { Flex } from "@chakra-ui/react"
import { urls } from "../globals/data"
import { SolidLink } from "./commons"

export function BreadCrumb({ project, ...props }) {
  return (
    <Flex alignItems="center" fontSize="sm" {...props}>
      <SolidLink to={urls.gallery} py={1} mr={2}>
        Home
      </SolidLink>{" "}
      /{" "}
      <SolidLink to={project.links.app} py={1} ml={2}>
        {project.title}
      </SolidLink>
    </Flex>
  )
}
