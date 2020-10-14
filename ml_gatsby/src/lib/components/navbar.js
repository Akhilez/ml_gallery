import React from "react"
import ak_logo from "src/lib/media/ak_logo.png"
import { urls } from "src/lib/globals/data"
import { Box, Flex, Image } from "@chakra-ui/core"
import { Link as GLink } from "gatsby"
import { FaGithub, FiMenu, MdHome, MdPerson } from "react-icons/all"
import { Container, SolidLink } from "./commons"

function NavItem({ href, text, icon, ...props }) {
  return (
    <SolidLink href={href} fontSize="sm" {...props}>
      <Flex alignItems="center">
        <Box as={icon} fontSize="lg" mr={2} />
        {text}
      </Flex>
    </SolidLink>
  )
}

export default function Navbar() {
  const [show, setShow] = React.useState(false)

  return (
    <Container>
      <Flex as="nav" alignItems="center" justify="space-between" wrap="wrap">
        <GLink to={urls.profile} className="navbar-brand logo">
          <Image src={ak_logo} height="40px" alt="ak_logo" ml={2} />
        </GLink>

        <Box
          display={{ base: "block", sm: "none" }}
          onClick={() => setShow(!show)}
        >
          <FiMenu />
        </Box>

        <Box
          display={{ base: show ? "block" : "none", sm: "flex" }}
          width={{ base: "full", sm: "auto" }}
        />

        <Box
          display={{ base: show ? "block" : "none", sm: "flex" }}
          mt={{ base: 4, sm: 0 }}
          bg="backgroundColor"
          zIndex={5}
        >
          <NavItem href={urls.gallery} text="ML GALLERY" icon={MdHome} />
          <NavItem href={urls.profile} text="PROFILE" icon={MdPerson} />
          <NavItem href={urls.repo} text="REPO" icon={FaGithub} isExternal />
        </Box>
      </Flex>
    </Container>
  )
}
