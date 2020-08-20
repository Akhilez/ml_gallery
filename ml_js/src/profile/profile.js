import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import ProfileNavBar from "./navbar";
import {
  ProfileBadge,
  GithubCalendar,
  Social,
  ResumeButton,
  ProjectBox,
} from "./profile_components";
import toggle_img from "./media/toggle.png";
import "./css/timeline.css";
import code_art_img from "./media/cover_code_art_with_bg_dark.png";
import neuralhack from "./media/neuralhack.jpg";
import revolutionUC from "./media/revolutionuc.jpg";
import projects from "./data/projects";
import atheism_img from "./media/misc/evolution.jpg";
import vegan_img from "./media/misc/chicken.jpg";
import kmitra_img from "./media/misc/kmitraLogo.jpg";
import ezio_img from "./media/misc/ezio.jpg";
import mlg_img from "../landing/ml_logo/ml_logo.png";
import { Link } from "react-router-dom";
import urls from "../urls";
import { Helmet } from "react-helmet";
import vndly_logo from "./media/timeline/vndly_logo.png";
import uc_logo from "./media/timeline/uc.png";
import aviso_logo from "./media/timeline/aviso.png";
import kmit_logo from "./media/timeline/kmit.jpg";

const profilePhoto = "/media/profile_photo.jpg";

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
      "Deep Learning Researcher. Master's in AI 🎓. Neural Nets 🧠, Web 🖥, Mobile 📱, Cloud ☁️, UI.";
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
    const Emoji = (props) => (
      <span className="emoji_text">{props.children}</span>
    );

    return (
      <div>
        <p className="para no_href_p">
          I am obsessed with
          <b>
            <i>Deep Learning</i>
          </b>
          <Emoji>🧠</Emoji>, <i>Productivity</i> <Emoji>👨🏻‍💻</Emoji> and
          <i>Space Exploration</i> <Emoji>🪐</Emoji>
        </p>

        <div className="header1" style={{ fontSize: 20, paddingTop: 20 }}>
          <img src={toggle_img} alt="toggle" height="30px" />
          &nbsp; Available for hire.
        </div>
      </div>
    );
  }

  TimeLine() {
    const TimeLineItem = ({
      date,
      image,
      brand,
      role,
      description,
      linkRole,
      linkBrand,
    }) => {
      return (
        <li className="event" data-date={date}>
          <Row>
            <Col sm="auto">
              <img
                width="50px"
                src={image}
                alt={brand}
                style={{ marginTop: 25, borderRadius: 4 }}
              />
            </Col>
            <Col>
              <h3 className={"timeline_heading"}>
                {linkRole ? <a href={linkRole}>{role}</a> : role}
              </h3>
              <p>{linkBrand ? <a href={linkBrand}>{brand}</a> : brand}</p>
              {description}
            </Col>
          </Row>
        </li>
      );
    };
    return (
      <div>
        <h3 className="header1">Timeline</h3>

        <div id="timeline_section">
          <ul className="timeline no_href">
            <TimeLineItem
              date="2020"
              image={vndly_logo}
              brand="VNDLY"
              role="Deep Learning Engineer"
              linkBrand="https://vndly.com/"
              description={
                <p>
                  Boosted the accuracy of production-grade Deep-Learning model
                  based on
                  <b>
                    <i> Google's BERT </i>
                  </b>
                  for an NLP task of matching candidates to job descriptions.
                  Technologies:
                  <i>PyTorch, TensorFlow, Django, React</i>
                  <br />
                  Designed an automated training pipeline for
                  <strong>active learning</strong>
                </p>
              }
            />
            <TimeLineItem
              date="2019"
              image={uc_logo}
              brand="University of Cincinnati"
              role="Master’s in Artificial Intelligence"
              linkBrand="https://www.uc.edu/"
              linkRole="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/detail/20MAS-AI-MENG"
              description={
                <p>
                  Specializations:
                  <i>
                    Computer Vision, NLP, Reinforcement Learning and Complex
                    Intelligent Systems.
                  </i>
                </p>
              }
            />
            <TimeLineItem
              date="2018"
              image={aviso_logo}
              brand="Aviso AI"
              role="Full-Stack Engineer"
              linkBrand="https://www.aviso.com/"
              description={
                <p>
                  Reduced the ML cloud cost by
                  <b>
                    <i>60%</i>
                  </b>
                  <br />
                  Technologies used:
                  <i>Django, Kubernetes, AWS, Linux, Puppet, Vue.js</i>
                </p>
              }
            />
            <TimeLineItem
              date="2015"
              image={kmit_logo}
              brand="Keshav Memorial Institute of Technology"
              role="Part-Time Developer"
              linkBrand="https://kmit.in/home"
              description={
                <p>
                  Developed apps for the college’s operations like
                  <a href="http://akhilez.com/home/all_projects//#student_feedback">
                    Student Feedback
                  </a>
                  and
                  <a href="http://akhilez.com/home/all_projects//#gatepass">
                    Gate-Pass System
                  </a>
                </p>
              }
            />

            <TimeLineItem
              date="2014"
              image={kmit_logo}
              brand="Keshav Memorial Institute of Technology"
              role="Bachelor's in Computer Science and Engineering"
              linkBrand="https://kmit.in/home"
            />
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
            <Link to={urls.ml_gallery.url}>
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
        I absolutely ❤ coding! Each green box below represents the amount of
        coding on that day of the year.
        <GithubCalendar />
        <div className="row">
          <img src={code_art_img} alt="CoverPhoto" width="400" />
        </div>
      </div>
    );
  }

  Achievements(props) {
    return (
      <div>
        <h3 className="header1">Hackathons</h3>

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
              Won
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
              I built an Augmented Reality game called
              <a href="http://akhilez.com/home/all_projects//#alster">Alster</a>
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
            Andrej is very special to me. He was just a normal
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.youtube.com/user/badmephisto"
            >
              youtuber
            </a>
            who taught me rubik’s cube with his
            <a
              target="_blank"
              rel="noopener noreferrer"
              href="https://www.youtube.com/user/badmephisto"
            >
              videos
            </a>
            . But he became very successful as I saw him grow older. Today he is
            the director of AI at Tesla. My career path is a huge inspiration
            from his career path. Even this website is lightly inspired from his
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
            The most influential professor of my life. He is the reason why I
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
            <a href="mailto: akhilez.ai@gmail.com"> akhilez.ai@gmail.com</a>
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
