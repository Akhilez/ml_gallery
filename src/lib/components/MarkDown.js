import React from "react"
import ReactMarkdown from "react-markdown"
import gfm from "remark-gfm"
import ChakraUIRenderer from "chakra-ui-markdown-renderer"
import { Box } from "@chakra-ui/react"

export const MD = ({ source, ...props }) => (
  <Box
    as={ReactMarkdown}
    plugins={[gfm]}
    renderers={ChakraUIRenderer()}
    escapeHtml={false}
    source={source}
    props={props}
  />
)
