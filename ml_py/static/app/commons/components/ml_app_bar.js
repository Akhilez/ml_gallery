import React from "react";
import Navbar from "react-bootstrap/Navbar";
import Nav from "react-bootstrap/Nav";
import ak_logo from '../images/AK_logo.svg';
import '../../landing/landing.css';
import urls from "../../urls.json";
import {Helmet} from "react-helmet";
import ml_logo from '../../landing/ml_logo/ml_logo.png';
import { GitHub, Home, Person } from '@material-ui/icons';
import './components.css';


class MLAppBar extends React.Component {
    render() {
        return (
            <Navbar bg="transparent" variant="light">
                <Nav className="mr-auto">
                    <Navbar.Brand href="/profile"><img src={ak_logo} alt={"AK Logo"} height={"40px"}/></Navbar.Brand>
                </Nav>
                <Nav.Link href={urls.ml_gallery.url} className={"nav-link"}>
                    <div><Home fontSize={"small"} className={"navIcon"}/> HOME</div>
                </Nav.Link>
                <Nav.Link href={urls.profile.url} className={"nav-link"}>
                    <div><Person fontSize={"small"} className={"navIcon"}/> PROFILE</div>
                </Nav.Link>
                <Nav.Link href="https://github.com/Akhilez/ml_gallery" className={"nav-link"} target={"_blank"}>
                    <div><GitHub fontSize={"small"} className={"navIcon"}/> REPO</div>
                </Nav.Link>
                <this.metaTags/>
            </Navbar>
        );
    }

    metaTags(props) {
        let desc = "Machine Learning Gallery is a master project of deep learning tasks involving Computer Vision, Natural Language Processing, Reinforcement Learning and Unsupervised Learning with visualizations and explanations. Developed by Akhilez";
        let title = "Machine Learning Gallery | Akhilez";
        return (
            <Helmet>
                <meta name="description"
                      content={desc}/>

                <meta name="twitter:image:src" content={ml_logo}/>
                <meta name="twitter:site" content="@akhilez_"/>
                <meta name="twitter:creator" content="@akhilez_"/>
                <meta name="twitter:card" content="summary_large_image"/>
                <meta name="twitter:title" content={title}/>
                <meta name="twitter:description" content={desc}/>

                <meta property="og:image" content={ml_logo}/>
                <meta property="og:site_name" content={title}/>
                <meta property="og:type" content="object"/>
                <meta property="og:title" content={title}/>
                <meta property="og:url" content="https://akhil.ai/gallery"/>
                <meta property="og:description"
                      content={desc}/>

            </Helmet>
        );
    }
}

export default MLAppBar;
