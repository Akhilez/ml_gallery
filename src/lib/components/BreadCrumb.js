import React from "react"
import { Flex, Link, useColorModeValue } from "@chakra-ui/react"
import { urls } from "../globals/data"
import { Link as GLink } from "gatsby"

const BreadCrumbLink = ({ href, ...props }) => (
  <Link
    as={GLink}
    to={href}
    py={1}
    href={href}
    _hover={{
      color: "brand.500",
      textDecoration: "none",
    }}
    mr={2}
    {...props}
  >
    {props.children}
  </Link>
)

export function BreadCrumb({ project, ...props }) {
  const color = useColorModeValue("gray.500", "gray.300")
  return (
    <Flex alignItems="center" color={color} fontSize="sm" {...props}>
      <BreadCrumbLink to={urls.gallery}>Home</BreadCrumbLink> /
      <BreadCrumbLink to={project.links.app} ml={2}>
        {project.title}
      </BreadCrumbLink>
    </Flex>
  )
}
