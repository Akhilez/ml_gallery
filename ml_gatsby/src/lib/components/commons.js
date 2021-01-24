import React from "react"
import {
  Box,
  Divider,
  Link,
  Text,
  Flex,
  Image,
  IconButton,
  useColorModeValue,
  useTheme,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import { urls } from "../globals/data"
import ak_logo_white from "src/lib/media/ak_white.svg"
import { AiFillGithub, BiEnvelope } from "react-icons/all"

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

const ContactIcon = ({ icon: Icon, url }) => (
  <IconButton as={Link} mx={2} isRound size="sm" href={url} isExternal>
    <Icon />
  </IconButton>
)

export function Footer() {
  const bg = useColorModeValue("gray.600", "gray.900")
  return (
    <Box backgroundColor={bg} h="150px">
      <Container pt="50px" pl={4}>
        <Flex justify="space-between">
          <Flex align="center">
            <Image src={ak_logo_white} height="20px" />
            <Text color="white" ml={4} fontSize="20px">
              ML Gallery
            </Text>
          </Flex>
          <Flex>
            <ContactIcon icon={AiFillGithub} url={urls.repo} />
            <ContactIcon icon={BiEnvelope} url="mailto:akhilez.ai@gmail.com" />
          </Flex>
        </Flex>
      </Container>
    </Box>
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
