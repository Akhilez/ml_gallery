import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProfileNavBar from "./navbar";
import {
  ProfileBadge,
  MyGithubCalendar,
  Social,
  ResumeButton,
  ProjectBox,
} from "./profile_components";
import toggle_img from "./media/toggle.png";
import code_art_img from "./media/cover_code_art_with_bg_dark.png";
import neuralhack from "./media/neuralhack.jpg";
import revolutionUC from "./media/revolutionuc.jpg";
import projects from "./data/projects";
import atheism_img from "./media/evolution.jpg";
import vegan_img from "./media/chicken.jpg";
import kmitra_img from "./media/kmitraLogo.jpg";
import ezio_img from "./media/ezio.jpg";
import mlg_img from "../gallery/ml_logo/ml_logo.png";
import urls from "../../data/urls.json";
import { Helmet } from "react-helmet";
import profilePhoto from "./media/profile_photo.jpg";
import Link from "next/link";

export default class ProfilePage extends React.Component {
  render() {
    return (
      <div className={"profile_root"}>
        <Container>
          <this.metaTags />
          <ProfileNavBar active={"profile"} />
          <ProfileBadge />
          <this.Bio />
          <this.TimeLine />
          <this.DeepLearning />
          <this.CodingActivity />
          <this.Achievements />
          <this.Projects />
          <this.Misc />
          <this.Influencers />
          <this.Footer />
        </Container>
      </div>
    );
  }

  metaTags(props) {
    let desc =
      'Machine Learning Engineer. Master\'s in AI. Neural Nets, Web, Mobile, Cloud, UI. "Code is Art" - Akhilez';
    let title = "Akhil D. (Akhilez)";
    return (
      <Helmet>
        <meta name="description" content={desc} />

        <meta name="twitter:image:src" content={profilePhoto} />
        <meta name="twitter:site" content="@akhilez_" />
        <meta name="twitter:creator" content="@akhilez_" />
        <meta name="twitter:card" content="summary_large_image" />
        <meta name="twitter:title" content={title} />
        <meta name="twitter:description" content={desc} />

        <meta property="og:image" content={profilePhoto} />
        <meta property="og:site_name" content={title} />
        <meta property="og:type" content="object" />
        <meta property="og:title" content={title} />
        <meta property="og:url" content="https://akhil.ai" />
        <meta property="og:description" content={desc} />
      </Helmet>
    );
  }

  Bio(props) {
    return (
      <div>
        <h3 className="header1">Bio</h3>
        <p className="para no_href_p">
          I majored in <i>Artificial Intelligence</i> in my Master’s from{" "}
          <a href="https://www.uc.edu/">University of Cincinnati</a>,
          specialized in Deep Learning architectures for{" "}
          <i>
            Computer Vision, NLP, Reinforcement Learning and Complex Intelligent
            Systems
          </i>
          . Previously, I worked at an AI based startup called{" "}
          <a href="https://aviso.ai/">Aviso.AI</a> as a{" "}
          <i>Full-Stack Developer</i> with technologies -{" "}
          <i>Python, AWS and Vue.js</i>. During my undergrad, I worked as a
          part-time Software Developer at the college’s administrative
          department where I developed software applications for digitalization
          and automation of the administrative operations.
        </p>
        <p className="para no_href_p">
          I am extremely passionate about modern Artificial Intelligence. In my
          spare time, I try to recreate famous research works in deep learning
          and deploy them with user interaction at{" "}
          <Link href={urls.ml_gallery.url}>akhil.ai/gallery</Link>. I also work
          on my own independent projects. I developed a number of applications
          for the web and mobile over the years because I enjoy coding and
          designing. I associate my long-term goals strongly with pioneering the
          advancements in <i>Artificial General Intelligence</i> for further
          space exploration and more.
        </p>

        <div className="header1" style={{ fontSize: 20, paddingTop: 20 }}>
          <img src={toggle_img} alt="toggle" height="30px" />
          &nbsp; Available for hire.
        </div>
      </div>
    );
  }

  TimeLine(props) {
    return (
      <div>
        <h3 className="header1">Timeline</h3>

        <div id="timeline_section">
          <ul className="timeline no_href">
            <li className="event" data-date="2019">
              <h3 className={"timeline_heading"}>Python Developer</h3>
              <p>
                <a href="https://vndly.com/">VNDLY</a>
              </p>
              <p>
                Developed a Deep Learning model based on Google BERT for an NLP
                task of matching candidates to job descriptions.
              </p>
            </li>
            <li className="event" data-date="2019">
              <h3 className={"timeline_heading"}>
                <a href="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/detail/20MAS-AI-MENG">
                  Master’s in Artificial Intelligence
                </a>
              </h3>
              <p>
                <a href="https://www.uc.edu/">University of Cincinnati</a>
              </p>
              <p>
                <a href="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/majormap/20MAS-AI-MENG">
                  Courses taken:
                </a>{" "}
                Intelligent Systems, ML, AI, Deep Learning, Complex Systems,
                Computer Vision, StartupUC
              </p>
            </li>
            <li className="event" data-date="2018">
              <h3 className={"timeline_heading"}>Full-Stack Engineer</h3>
              <p>
                <a href="https://www.aviso.com/">Aviso Inc.</a>
              </p>
              <p>
                Worked on a wide variety of tasks revolving around the cloud
                infrastructure of the Aviso AI product.
              </p>
            </li>
            <li className="event" data-date="2015">
              <h3 className={"timeline_heading"}>Part-Time Developer</h3>
              <p>
                <a href="https://kmit.in/home">
                  Keshav Memorial Institute of Technology
                </a>
              </p>
              <p>
                Developed apps for the college’s operations like{" "}
                <a href="http://akhilez.com/home/all_projects//#student_feedback">
                  Student Feedback
                </a>{" "}
                and{" "}
                <a href="http://akhilez.com/home/all_projects//#gatepass">
                  Gate-Pass System
                </a>
              </p>
            </li>
            <li className="event" data-date="2014">
              <h3 className={"timeline_heading"}>
                Bachelor's in Computer Science and Engineering
              </h3>
              <p>
                <a href="https://kmit.in/home">
                  Keshav Memorial Institute of Technology
                </a>
              </p>
            </li>
          </ul>
        </div>
      </div>
    );
  }

  DeepLearning(props) {
    let topics = [
      {
        title: "Computer Vision",
        projects:
          "MNSIT GAN (ACGAN), Colorizer, Find All Numbers (Faster-RCNN)",
      },
      {
        title: "Natural Language Processing",
        projects:
          "Next Char (LSTM), Next Word (word2vev), Next Sentence (seq2seq with attention), BERT",
      },
      {
        title: "Reinforcement Learning",
        projects: "Policy Gradients - TicTacToe, Pong, Racer",
      },
      {
        title: "Unsupervised Learning",
        projects:
          "MNIST AutoEncoder, Self-Organizing Feature-Maps, Associative Memory",
      },
      {
        title: "Other",
        projects: "Spiking Neurons, MNIST Detection Dataset",
      },
    ];
    return (
      <div>
        <h3 className="header1">Deep Learning</h3>

        <div
          className="row project_box"
          style={{ marginBottom: -40, marginTop: -30 }}
        >
          <div className="col-md-3">
            <Link href={urls.ml_gallery.url}>
              <img
                src={mlg_img}
                className="project_image"
                alt={"MLGallery Logo"}
                width={"250px"}
                style={{ marginTop: 15 }}
              />
            </Link>
          </div>
          <div className="col-md-9">
            <p>
              My passion for deep learning started when I learned CNNs in 2016 -
              the booming period of deep learning. Since then, I have been
              experimenting with Neural Nets in my pet-projects, earned a
              Master’s degree in Artificial Intelligence and now, I’m developing
              a curation of interesting deep learning tasks (listed below) into
              a master project called Machine Learning Gallery
            </p>
          </div>
        </div>

        <div>
          {topics.map((topic) => (
            <Row key={topic.title}>
              <Col sm={"3"} style={{ fontWeight: 400 }}>
                {topic.title}
              </Col>
              <Col sm={"9"}>{topic.projects}</Col>
            </Row>
          ))}
        </div>

        <a
          href={urls.ml_gallery.url}
          className="btn btn-outline-secondary btn-lg resume-button"
          style={{ width: 230, marginTop: 30 }}
        >
          VISIT
        </a>
      </div>
    );
  }

  CodingActivity(props) {
    return (
      <div>
        <h3 className="header1 no_href_p">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://github.com/Akhilez"
            style={{ fontWeight: 700 }}
          >
            Coding Activity
          </a>
        </h3>
        <br />
        The live GitHub contributions below show my commitment to writing code
        <MyGithubCalendar />
        <div className="row">
          <img src={code_art_img} alt="CoverPhoto" width="400" />
        </div>
      </div>
    );
  }

  Achievements(props) {
    return (
      <div>
        <h3 className="header1">Achievements</h3>

        <div className="row project_box">
          <div className="col-md-7">
            <img
              className="project_image"
              src={neuralhack}
              alt="syllabus"
              width="600px"
            />
          </div>
          <div className="col-md-5">
            <h4 className="project_title">Won NeuralHack</h4>
            <p className="project_description">
              NeuralHack is an India-wide hackathon with approximately 13,000
              participants conducted by Virtusa. The tasks to be completed in 24
              hours were:
            </p>
            <ul className="project_description">
              <li>
                Build a machine learning model that predicts a class label from
                the given dataset.
              </li>
              <li>
                Build an IoT device that measures the alcohol content from the
                air and sends a signal to the cloud on reaching a threshold.
              </li>
            </ul>

            <div className="row">
              <div className="col-auto project_date">16th November, 2017</div>
            </div>
          </div>
        </div>

        <div className="row project_box">
          <div className="col-md-7">
            <img
              className="project_image"
              src={revolutionUC}
              alt="syllabus"
              width="600px"
            />
          </div>
          <div className="col-md-5">
            <h4 className="project_title no_href">
              Won{" "}
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="https://revolutionuc.com/"
              >
                RevolutionUC
              </a>
            </h4>
            <p className="project_description no_href_p">
              <a href="https://revolutionuc.com/">RevolutionUC</a> is a
              hackathon conducted by ACM at University of Cincinnati with
              roughly 400 participants.
            </p>
            <p className="project_description no_href_p">
              I built an Augmented Reality game called{" "}
              <a href="http://akhilez.com/home/all_projects//#alster">Alster</a>{" "}
              in 24 hours and won top 5.
            </p>

            <div className="row">
              <div className="col-auto project_date">22nd February, 2020</div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  Projects(props) {
    let profile_projects = projects.projects.filter(
      (project) => project.on_profile
    );
    return (
      <div>
        <h3 className="header1 no_href_p">
          <a
            href="https://github.com/Akhilez?tab=repositories"
            style={{ fontWeight: 700 }}
          >
            Independent Projects
          </a>
        </h3>

        <div>
          {profile_projects.map((project) => (
            <ProjectBox data={project} key={project.title} />
          ))}
          <a
            href={urls.all_projects.url}
            className="btn btn-outline-secondary btn-lg resume-button"
            style={{ width: 200 }}
          >
            SHOW MORE
          </a>
        </div>
      </div>
    );
  }

  Misc(props) {
    return (
      <div>
        <h3 className="header1">Misc</h3>

        <div className="row project_box">
          <div className="col-md-3">
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="http://kmit.in/emagazine/article/8934/"
            >
              <img
                className="project_image round-frame"
                src={kmitra_img}
                alt="kmitra"
                width="200px"
                height="200px"
              />
            </a>
          </div>
          <div className="col-md-9">
            <h4 className="project_title no_href">
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="http://kmit.in/emagazine/article/8934/"
              >
                A short story on AI
              </a>
            </h4>
            <p className="project_description no_href_p">
              In my undergrad, I was a monthly writer at the college e-magazine
              called
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="http://kmit.in/emagazine/author/akhil-kanna/"
              >
                kMITRA
              </a>
              . One of the articles I wrote is an interesting short story on AI
              called
              <a
                target="_blank"
                rel="noopener noreferrer"
                href="http://kmit.in/emagazine/article/8934/"
              >
                “PrecArIous Love”
              </a>
              .
            </p>
          </div>
        </div>

        <div className="row project_box">
          <div className="col-md-3">
            <img
              className="project_image round-frame"
              src={atheism_img}
              alt="evolution"
              width="200px"
              height="200px"
            />
          </div>
          <div className="col-md-9">
            <h4 className="project_title no_href">
              Atheism &amp; Rational Thought
            </h4>
            <p className="project_description no_href_p">
              I am an atheist. For me, atheism is not just denying God’s
              existence. That is easy. But I think atheism is an emergent
              phenomenon of rational thought. I encourage people to think
              scientifically and make logical decisions rather than following a
              herd.
            </p>
          </div>
        </div>

        <div className="row project_box">
          <div className="col-md-3">
            <img
              className="project_image round-frame"
              src={vegan_img}
              alt="chicken"
              width="200px"
              height="200px"
            />
          </div>
          <div className="col-md-9">
            <h4 className="project_title no_href">Being Vegan</h4>
            <p className="project_description no_href_p">
              I love animals. I became a vegetarian when I was 10 after
              witnessing an animal slaughter for that night’s dinner. I believe
              no human has moral rights to kill another conscious living thing
              unless it is life-threatening. Recently I stopped consuming all
              animal related products because it involves animal abuse to an
              unknown non-zero degree.
            </p>
          </div>
        </div>

        <div className="row project_box">
          <div className="col-md-3">
            <img
              className="project_image round-frame"
              src={ezio_img}
              alt="ezio"
              width="200px"
              height="200px"
            />
          </div>
          <div className="col-md-9">
            <h4 className="project_title no_href">My Sketches</h4>
            <p className="project_description no_href_p">
              I am very good at sketching. But I don’t find enough time and
              motivation to do it often. I’ll soon post a gallery of my
              sketches.
            </p>
          </div>
        </div>
      </div>
    );
  }

  Influencers(props) {
    return (
      <div className="project_box">
        <h5 className="project_title">Most influential people in my life</h5>
        <div className="inspiration_person_title no_href">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://twitter.com/elonmusk"
          >
            Elon Musk
          </a>
          <div className="inspiration_person_description">
            "Work every waking hour."
          </div>
        </div>
        <div className="inspiration_person_title no_href">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://twitter.com/neiltyson"
          >
            Neil DeGrasse Tyson
          </a>
          <div className="inspiration_person_description">
            "Science is true whether or not you believe in it."
          </div>
        </div>
        <div className="inspiration_person_title no_href">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://twitter.com/GrantCardone"
          >
            Grant Cardone
          </a>
          <div className="inspiration_person_description">
            “Stay dangerous” from “Be obsessed or be average”
          </div>
        </div>
        <div className="inspiration_person_title no_href">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://twitter.com/RGVzoomin"
          >
            Ram Gopal Varma
          </a>
          <div className="inspiration_person_description">
            “Naa ishtam” (translation: I decide what I do.)
          </div>
        </div>
        <div className="inspiration_person_title">
          <div className="no_href">
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://cs.stanford.edu/people/karpathy/"
            >
              Andrej Karpathy
            </a>
          </div>
          <div className="inspiration_person_description no_href_p">
            Andrej is very special to me. He was just a normal{" "}
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.youtube.com/user/badmephisto"
            >
              youtuber
            </a>{" "}
            who taught me rubik’s cube with his{" "}
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.youtube.com/user/badmephisto"
            >
              videos
            </a>
            . But he became very successful as I saw him grow older. Today he is
            the director of AI at Tesla. My career path is a huge inspiration
            from his career path. Even this website is lightly inspired from his{" "}
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://cs.stanford.edu/people/karpathy/"
            >
              website.
            </a>
          </div>
        </div>
        <div className="inspiration_person_title no_href">
          <a
            target="_blank"
            rel="noopener noreferrer"
            href="https://eecs.ceas.uc.edu/~aminai/"
          >
            Ali Minai
          </a>
          <div className="inspiration_person_description">
            The most influential professor in my life. He is the reason why I
            love academia so much.
          </div>
        </div>
      </div>
    );
  }

  Footer(props) {
    return (
      <footer>
        <div className="footer">
          <hr />

          <div style={{ marginTop: 15 }} className="roboto-light-ak no_href">
            <a href="mailto: ak@akhil.ai"> ak@akhil.ai</a>
          </div>
          <Social />

          <ResumeButton />
          <br />

          <p className="roboto-light-ak">Akhil Devarashetti</p>
        </div>
      </footer>
    );
  }
}
