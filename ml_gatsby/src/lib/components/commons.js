import React from "react"
import { Divider, Link, Text, useTheme } from "@chakra-ui/core"
import Box from "@chakra-ui/core/dist/Box"
import { Link as GLink } from "gatsby"

export const Container = ({ children, ...props }) => {
  const theme = useTheme()

  console.log(theme.breakpoints)
  console.log("slice")
  console.log(theme.breakpoints.slice(1))

  return (
    <Box
      {...props}
      mx="auto"
      maxW={["full", "full", ...theme.breakpoints.slice(1)]}
      w="100%"
    >
      {children}
    </Box>
  )
}

export function Footer() {
  return (
    <Container>
      <Divider />
      <Text>ML Gallery</Text>
    </Container>
  )
}

export function SolidLink({ href, ...props }) {
  const theme = useTheme()
  return (
    <Link
      as={GLink}
      to={href}
      py={2}
      px={3}
      href={href}
      fontSize="sm"
      display="block"
      borderRadius="lg"
      _hover={{
        color: "white",
        textDecoration: "none",
        backgroundColor: theme.colors.brand["500"],
        transitionDuration: "0.4s",
        marginLeft: "4px",
      }}
      {...props}
    >
      {props.children}
    </Link>
  )
}
