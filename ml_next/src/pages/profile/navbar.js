import React from "react";
import ak_logo from "./media/ak_logo.svg";
import urls from "../../data/urls.json";
import { Box, Link, Flex } from "@chakra-ui/core";

function NavItem({ href, text }) {
  return (
    <Link
      href={href}
      p={2}
      fontSize="sm"
      display="block"
      _hover={{ color: "white", textDecoration: "none" }}
    >
      {text}
    </Link>
  );
}

export default function ProfileNavBar(props) {
  const [show, setShow] = React.useState(false);

  return (
    <>
      <Flex
        as="nav"
        alignItems="center"
        justify="space-between"
        wrap="wrap"
        padding="4"
      >
        <a href={urls.profile.url} className="navbar-brand logo">
          <img src={ak_logo} width={"30px"} alt={"ak_logo"} />
        </a>

        <Box
          display={{ base: "block", sm: "none" }}
          onClick={() => setShow(!show)}
        >
          menu
        </Box>

        <Box
          display={{ base: show ? "block" : "none", sm: "flex" }}
          width={{ base: "full", sm: "auto" }}
        />

        <Box
          display={{ base: show ? "block" : "none", sm: "flex" }}
          mt={{ base: 4, sm: 0 }}
        >
          <NavItem href={urls.profile.url} text="PROFILE" />
          <NavItem href={urls.ml_gallery.url} text="ML GALLERY" />
          <NavItem href={urls.all_projects.url} text="PROJECTS" />
          <NavItem href={urls.resume.url} text="RESUME" />
        </Box>
      </Flex>
    </>
  );
}
