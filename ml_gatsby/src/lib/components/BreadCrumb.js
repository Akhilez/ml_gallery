import React from "react"
import { Text } from "@chakra-ui/core"
import { Link } from "gatsby"
import { urls } from "../globals/data"

export function BreadCrumb({ project }) {
  return (
    <Text>
      <Link to={urls.gallery}>Home</Link> >{" "}
      <Link to={project.links.app}>{project.title}</Link>
    </Text>
  )
}
