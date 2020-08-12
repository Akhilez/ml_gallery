import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import ak_logo from "../pages/gallery/images/AK_logo.svg";
import urls from "../data/urls.json";
import { Helmet } from "react-helmet";
import { FaGithub } from "react-icons/fa";
import { AiFillHome } from "react-icons/ai";
import { MdPerson } from "react-icons/md";

const ml_logo = "/media/ml_logo.png";

class MLAppBar extends React.Component {
  render() {
    return (
      <Navbar bg="transparent" variant="light">
        <Nav className="mr-auto">
          <Navbar.Brand href="/profile">
            <img src={ak_logo} alt={"AK Logo"} height={"40px"} />
          </Navbar.Brand>
        </Nav>
        <Nav.Link href={urls.ml_gallery.url} className={"nav-link"}>
          <div>
            <AiFillHome fontSize={"small"} className={"navIcon"} /> HOME
          </div>
        </Nav.Link>
        <Nav.Link href={urls.profile.url} className={"nav-link"}>
          <div>
            <MdPerson fontSize={"small"} className={"navIcon"} /> PROFILE
          </div>
        </Nav.Link>
        <Nav.Link
          href="https://github.com/Akhilez/ml_gallery"
          className={"nav-link"}
          target={"_blank"}
        >
          <div>
            <FaGithub fontSize={"small"} className={"navIcon"} /> REPO
          </div>
        </Nav.Link>
        <this.metaTags />
      </Navbar>
    );
  }

  metaTags(props) {
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
