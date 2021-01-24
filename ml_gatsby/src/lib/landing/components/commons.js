import { Image, Link, Wrap, WrapItem } from "@chakra-ui/react"
import { BiCode } from "react-icons/all"
import colabIcon from "../images/colab.png"
import React from "react"

export const IconLinks = ({ project }) => (
  <Wrap mt={4}>
    {project.links.source && (
      <WrapItem>
        <Link
          as={BiCode}
          fontSize="20px"
          color="secondary.500"
          href={project.links.source}
          isExternal
        />
      </WrapItem>
    )}
    {project.links.colab && (
      <WrapItem>
        <Link
          as={Image}
          src={colabIcon}
          height="20px"
          href={project.links.colab}
          isExternal
        />
      </WrapItem>
    )}
  </Wrap>
)
