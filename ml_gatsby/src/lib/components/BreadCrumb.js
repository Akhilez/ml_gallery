import React from "react"
import { Flex } from "@chakra-ui/core"
import { urls } from "../globals/data"
import { SolidLink } from "./commons"

export function BreadCrumb({ project }) {
  return (
    <Flex alignItems="center" fontSize="sm">
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
