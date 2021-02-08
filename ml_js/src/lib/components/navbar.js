import React from "react"
import ak_logo from "src/lib/media/ak_logo.png"
import { urls } from "src/lib/globals/data"
import {
  Box,
  Flex,
  Image,
  useColorMode,
  Icon,
  useColorModeValue,
  Container,
} from "@chakra-ui/react"
import { Link as GLink } from "gatsby"
import {
  FaGithub,
  FiMenu,
  MdHome,
  MdPerson,
  IoMdMoon,
  IoMdSunny,
} from "react-icons/all"
import { SolidLink } from "./commons"

const NavLink = ({ href, icon, ...props }) => {
  const color = useColorModeValue("brand.400", "gray.400")
  return (
    <SolidLink href={href} color={color} mx={1} {...props}>
      <Icon as={icon} fontSize="lg" />
    </SolidLink>
  )
}

const DarkModeButton = () => {
  const { colorMode, toggleColorMode } = useColorMode()
  const isLight = colorMode === "light"
  const color = useColorModeValue("brand.400", "gray.400")
  return (
    <SolidLink as={Box} mx={1} color={color} onClick={toggleColorMode}>
      <Icon fontSize="lg" as={isLight ? IoMdMoon : IoMdSunny} />
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
          style={{ color: "gray.400" }}
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

export const StaticNavbar = () => {
  const bg = useColorModeValue("white", "gray.800")
  return (
    <Box backgroundColor={bg} position="sticky" top="0" py={2} zIndex={5}>
      <Container>
        <Flex as="nav" alignItems="center" justify="space-between" wrap="wrap">
          <GLink to={urls.profile} className="navbar-brand logo">
            <Image src={ak_logo} height="30px" alt="ak_logo" mt={1} />
          </GLink>
          <Box width="auto" />
          <Flex>
            <DarkModeButton />
            <NavLink href={urls.gallery} icon={MdHome} />
            <NavLink href={urls.profile} icon={MdPerson} />
            <NavLink href={urls.repo} icon={FaGithub} isExternal />
          </Flex>
        </Flex>
      </Container>
    </Box>
  )
}
