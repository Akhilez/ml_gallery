import React from "react"
import ak_logo from "src/lib/media/ak_logo.png"
import { urls } from "src/lib/globals/data"
import { Helmet } from "react-helmet"
import ml_logo from "src/lib/media/ml_logo/ml_logo.png"
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

export function MetaTags() {
  let desc =
    "Machine Learning Gallery is a master project of deep learning tasks involving Computer Vision, Natural Language Processing, Reinforcement Learning and Unsupervised Learning with visualizations and explanations. Developed by Akhilez"
  let title = "Machine Learning Gallery | Akhilez"
  return (
    <Helmet>
      <meta name="description" content={desc} />

      <meta name="twitter:image:src" content={ml_logo} />
      <meta name="twitter:site" content="@akhilez_" />
      <meta name="twitter:creator" content="@akhilez_" />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={desc} />

      <meta property="og:image" content={ml_logo} />
      <meta property="og:site_name" content={title} />
      <meta property="og:type" content="object" />
      <meta property="og:title" content={title} />
      <meta property="og:url" content="https://akhil.ai/gallery" />
      <meta property="og:description" content={desc} />
    </Helmet>
  )
}

export default function Navbar() {
  const [show, setShow] = React.useState(false)

  return (
    <Container>
      <MetaTags />
      <Flex as="nav" alignItems="center" justify="space-between" wrap="wrap">
        <GLink to={urls.profile} className="navbar-brand logo">
          <Image src={ak_logo} height="40px" alt="ak_logo" />
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
        >
          <NavItem href={urls.gallery} text="ML GALLERY" icon={MdHome} />
          <NavItem href={urls.profile} text="PROFILE" icon={MdPerson} />
          <NavItem href={urls.repo} text="REPO" icon={FaGithub} isExternal />
        </Box>
      </Flex>
    </Container>
  )
}
