import React from "react"
import ak_logo from "src/lib/media/ak_logo.png"
import { urls } from "src/lib/globals/data"
import { Box, Flex, Image, IconButton, useColorMode } from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import {
  FaGithub,
  FiMenu,
  MdHome,
  MdPerson,
  IoMdMoon,
  IoMdSunny,
} from "react-icons/all"
import { Container, SolidLink } from "./commons"

const NavLink = ({ href, icon, ...props }) => (
  <SolidLink href={href} mx={1} {...props}>
    <Box as={icon} fontSize="lg" />
  </SolidLink>
)

const DarkModeButton = () => {
  const { colorMode, toggleColorMode } = useColorMode()
  const isLight = colorMode === "light"
  return (
    <SolidLink as={Box} mx={1} onClick={toggleColorMode}>
      <Box fontSize="lg" as={isLight ? IoMdMoon : IoMdSunny} />
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
          zIndex={5}
        >
          <DarkModeButton />
          <NavLink href={urls.gallery} icon={MdHome} />
          <NavLink href={urls.profile} icon={MdPerson} />
          <NavLink href={urls.repo} icon={FaGithub} isExternal />
        </Box>
      </Flex>
    </Container>
  )
}
