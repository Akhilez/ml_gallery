import React from "react"
import { Box, Divider, Link, Text, useTheme } from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { urls } from "../globals/data"

export const Container = ({ children, ...props }) => {
  const theme = useTheme()

  return (
    <Box
      mx="auto"
      maxW={["full", "full", ...theme.breakpoints.slice(1)]}
      w="100%"
      {...props}
    >
      {children}
    </Box>
  )
}

export function Centered(props) {
  return <div align={"center"}>{props.children}</div>
}

export function Footer() {
  return (
    <Container textAlign="center" mt="50px">
      <Centered>
        <Divider />
        <Box my="25px">
          <Text>
            ML Gallery by{" "}
            <Link href={urls.profile} fontStyle="italic" fontWeight="bold">
              Akhilez
            </Link>
          </Text>
        </Box>
      </Centered>
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
      display="block"
      borderRadius="lg"
      _hover={{
        color: "white",
        textDecoration: "none",
        backgroundColor: theme.colors.brand["500"],
        transitionDuration: "0.4s",
      }}
      {...props}
    >
      {props.children}
    </Link>
  )
}
