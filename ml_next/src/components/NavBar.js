import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import NextLink from "next/link";
import ak_logo from "../pages/gallery/images/AK_logo.svg";
import urls from "../data/urls.json";
import { Helmet } from "react-helmet";
import { FaGithub } from "react-icons/fa";
import { AiFillHome } from "react-icons/ai";
import { MdPerson } from "react-icons/md";
import { Flex, Box } from "@chakra-ui/core/dist";

const ml_logo = "/media/ml_logo.png";

function NavButton({ url, icon, text, isExternal }) {
  return (
    <Flex as={isExternal ? "a" : NextLink} href={url}>
      <Flex className={"nav-link"} alignItems="center">
        <Box as={icon} fontSize={"md"} mr={2} className={"navIcon"} /> {text}
      </Flex>
    </Flex>
  );
}

class MLAppBar extends React.Component {
  render() {
    return (
      <Navbar bg="transparent" variant="light">
        <Nav className="mr-auto">
          <Navbar.Brand href="/profile">
            <img src={ak_logo} alt={"AK Logo"} height={"40px"} />
          </Navbar.Brand>
        </Nav>
        <NavButton url={urls.ml_gallery.url} icon={AiFillHome} text="HOME" />
        <NavButton url={urls.profile.url} icon={MdPerson} text="PROFILE" />
        <NavButton
          url="https://github.com/Akhilez/ml_gallery"
          icon={FaGithub}
          text="REPO"
          isExternal
        />
        <this.metaTags />
      </Navbar>
    );
  }

  metaTags() {
    let desc =
      "Machine Learning Gallery is a master project of deep learning tasks involving Computer Vision, Natural Language Processing, Reinforcement Learning and Unsupervised Learning with visualizations and explanations. Developed by Akhilez";
    let title = "Machine Learning Gallery | Akhilez";
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
    );
  }
}

export default MLAppBar;
