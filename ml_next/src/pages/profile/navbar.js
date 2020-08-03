import { Nav, Navbar } from "rsuite";
import React from "react";
import ak_logo from "./media/ak_logo.svg";
import urls from "../../data/urls.json";

export class ProfileNavBar2 extends React.Component {
  render() {
    return (
      <div>
        <Navbar
          appearance="inverse"
          activekey={this.props.active}
          className={"profile_navbar"}
          style={{ backgroundColor: "transparent" }}
        >
          <Navbar.Header>
            <a href={urls.profile.url} className="navbar-brand logo">
              <img
                src={ak_logo}
                width={"30px"}
                alt={"ak_logo"}
                style={{ paddingTop: 10 }}
              />
            </a>
          </Navbar.Header>
          <Navbar.Body>
            <Nav pullRight onSelect={() => {}} activeKey={"activeKey"}>
              <Nav.Item href={urls.profile.url} eventKey="profile">
                PROFILE
              </Nav.Item>
              <Nav.Item href={urls.ml_gallery.url} eventKey="ai">
                ML GALLERY
              </Nav.Item>
              <Nav.Item href={urls.all_projects.url} eventKey="all_projects">
                PROJECTS
              </Nav.Item>
              <Nav.Item href={urls.resume.url} eventKey="resume">
                RESUME
              </Nav.Item>
            </Nav>
          </Navbar.Body>
        </Navbar>
      </div>
    );
  }
}

import { Box, Heading, Flex, Text, Button } from "@chakra-ui/core";

const MenuItems = ({ children }) => (
  <Text mt={{ base: 4, md: 0 }} mr={6} display="block">
    {children}
  </Text>
);

// Note: This code could be better, so I'd recommend you to understand how I solved and you could write yours better :)
export default function ProfileNavBar(props) {
  const [show, setShow] = React.useState(false);
  const handleToggle = () => setShow(!show);

  return (
    <>
      <Flex
        as="nav"
        align="center"
        justify="space-between"
        wrap="wrap"
        padding="1.5rem"
        bg="teal.500"
        color="white"
        {...props}
      >
        <Flex align="center" mr={5}>
          <Heading as="h1" size="lg" letterSpacing={"-.1rem"}>
            Chakra UI
          </Heading>
        </Flex>

        <Box display={{ base: "block", md: "none" }} onClick={handleToggle}>
          <svg
            fill="white"
            width="12px"
            viewBox="0 0 20 20"
            xmlns="http://www.w3.org/2000/svg"
          >
            <title>Menu</title>
            <path d="M0 3h20v2H0V3zm0 6h20v2H0V9zm0 6h20v2H0v-2z" />
          </svg>
        </Box>

        <Box
          display={{ sm: show ? "block" : "none", md: "flex" }}
          width={{ sm: "full", md: "auto" }}
          alignItems="center"
          flexGrow={1}
        >
          <MenuItems>Docs</MenuItems>
          <MenuItems>Examples</MenuItems>
          <MenuItems>Blog</MenuItems>
        </Box>

        <Box
          display={{ sm: show ? "block" : "none", md: "block" }}
          mt={{ base: 4, md: 0 }}
        >
          <Button bg="transparent" border="1px">
            Create account
          </Button>
        </Box>
      </Flex>
      <ProfileNavBar2 />
    </>
  );
}
