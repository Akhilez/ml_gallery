import React from "react";
import Row from "react-bootstrap/Row";
import { BsCode } from "react-icons/bs";
import colabImage from "./images/colab.png";
import { Centered } from "../../components/common";
import learn_line_image from "./images/learn_line.png";
import learn_curve_img from "./images/learn_curve.png";
import find_number_img from "./images/find_number.png";
import colorizer_img from "./images/colorizer.png";
import deep_iris_img from "./images/deep_iris.png";
import faster_rcnn_img from "./images/faster_rcnn.png";
import linear_classifier_img from "./images/linear_classifier.png";
import mnist_gan_img from "./images/mnist_gan.png";
import pong_img from "./images/pong.png";
import tictactoe_img from "./images/tictactoe.png";
import under_construction2_img from "./images/under_construction2.png";
import which_char_img from "./images/which_char.png";

export default class Project extends React.Component {
  render() {
    return (
      <div className={"ProjectContainer"}>
        <this.ProjectImage project={this.props.project} />

        <div className={"project-text-block"}>
          <h2 style={{ fontSize: 32 }}>
            <a className={"link"} href={this.props.project.links.app}>
              {this.props.project.title}
            </a>
          </h2>
          <p style={{ fontSize: 20 }}>{this.props.project.desc}</p>
          {/*this.props.project.status !== "done" && `status: ${this.props.project.status}`*/}
          {this.getIconLinks(this.props.project)}
        </div>
        {this.props.children !== null && <Row>{this.props.children}</Row>}
      </div>
    );
  }

  getIconLinks(project) {
    return (
      <div className={"row"}>
        {project.links.source && (
          <div className={"col-auto"}>
            <a className={"link"} href={project.links.app}>
              <BsCode />
            </a>
          </div>
        )}

        {project.links.colab && (
          <div
            className={"col-auto"}
            style={{
              backgroundImage: `url(${colabImage})`,
              backgroundPosition: "center",
              backgroundSize: "contain",
              backgroundRepeat: "no-repeat",
            }}
          >
            <a
              className={"link"}
              href={project.links.colab}
              target="_blank"
              rel="noopener noreferrer"
            >
              <div style={{ height: "28px", width: "40px" }} />
            </a>
          </div>
        )}
      </div>
    );
  }

  ProjectImage(props) {
    return (
      <Centered>
        <a href={props.project.links.app}>
          <img
            src={props.project.image}
            className={"project-image"}
            alt={props.project.title + "Image"}
          />
        </a>
      </Centered>
    );
  }
}

export const projects = {
  categories: [
    {
      title: "Feed-Forward Networks",
      projects: [
        {
          title: "Learn a Line",
          id: 1,
          desc:
            "Visualize the learning of a basic neural network by learning a straight line equation.",
          image: learn_line_image,
          status: "in_progress",
          links: {
            app: "/feed_forward/learn_line",
            source: "https://github.com/Akhilez/ml_gallery",
          },
        },
        {
          title: "Linear Classifier",
          id: 2,
          desc:
            "Visualize the learning of a linear classifier: learns to distinguish between two different type of points in space that are linearly separable.",
          image: linear_classifier_img,
          status: "in_progress",
          links: {
            app: "/linear_classifier",
            source: "https://github.com/Akhilez/ml_gallery",
          },
        },
        {
          title: "Learn A Curve!",
          id: 9,
          desc:
            "Neural Networks can learn any continuous function! Here's a model that visualizes this concept. You can add data points and the network will learn its mathematical function.",
          image: learn_curve_img,
          status: "in_progress",
          links: {
            app: "/learn_curve",
            source: "https://github.com/Akhilez/ml_gallery",
            colab: "https://github.com/Akhilez/ml_gallery",
          },
        },
        {
          title: "Deep Iris",
          id: 3,
          desc:
            "Visualize how a multi-layer neural network tries to classify flowers from its petal and sepal dimensions",
          image: deep_iris_img,
          status: "to_do",
          links: {
            app: "/deep_iris",
            source: "https://github.com/Akhilez/ml_gallery",
          },
        },
      ],
    },
    {
      title: "Computer Vision",
      projects: [
        {
          title: "Which Character?",
          id: 4,
          desc: "Visualize the convolutions in recognizing handwritten letters",
          image: which_char_img,
          status: "to_do",
          links: {
            app: "/which_letter",
            source: "https://github.com/Akhilez/ml_gallery",
          },
        },
        {
          title: "MNIST GAN",
          id: 5,
          desc:
            "Generate handwritten numbers using Generative Adversarial Network fused with a Classifier.",
          image: mnist_gan_img,
          status: "to_do",
          links: {
            app: "/mnist_gan",
            source: "/mnist_gan",
            colab:
              "https://colab.research.google.com/github/Akhilez/ml_gallery/blob/master/ml_py/MLGallery/vision/selective_generator/colab.ipynb",
          },
        },
        {
          title: "Colorizer",
          id: 6,
          desc:
            "A web application that colorizes grayscale images using a Convolutional Neural Network. This was my major project in my undergrad.",
          image: colorizer_img,
          status: "to_do",
          links: {
            app: "https://akhilez.com/colorizer/",
            source: "https://github.com/Akhilez/ml_gallery",
          },
        },
        {
          title: "Find The Number",
          id: 7,
          desc:
            "Find the location of all the numbers in a large box and recognize what numbers they are.",
          image: find_number_img,
          status: "to_do",
          links: {
            app: "/number_detector",
            source: "/number_detector",
          },
        },
        {
          title: "Find All Numbers: V1",
          id: 10,
          desc: "",
          image: faster_rcnn_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Find All Numbers: V2",
          id: 11,
          desc: "",
          image: faster_rcnn_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Attention, Attention!",
          id: 12,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Style, Please: V1",
          id: 13,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Style, Please: V2",
          id: 14,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
      ],
    },
    {
      title: "Natual Language Processing",
      projects: [
        {
          title: "Next Char",
          id: 15,
          desc:
            "Visualize how a Recurrent Neural Network predicts which letter comes next.",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/next_char",
            source: "/next_char",
          },
        },
        {
          title: "Word To Vector: V1",
          id: 16,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Next Word",
          id: 17,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Word To Vector: V2",
          id: 18,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Next Sentence",
          id: 19,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
      ],
    },
    {
      title: "Reinforcement Learning",
      projects: [
        {
          title: "TicTacToe",
          id: 20,
          desc: "",
          image: tictactoe_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Ping-Pong",
          id: 21,
          desc: "",
          image: pong_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Racer",
          id: 22,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
      ],
    },
    {
      title: "Unsupervised Learning",
      projects: [
        {
          title: "AutoEncoder: V1",
          id: 23,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "AutoEncoder: V2",
          id: 24,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "MNIST Feature Maps",
          id: 25,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "Memorize Please",
          id: 26,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
      ],
    },
    {
      title: "Miscellaneous",
      projects: [
        {
          title: "Spiking Neurons",
          id: 27,
          desc: "",
          image: under_construction2_img,
          status: "to_do",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
        {
          title: "MNIST Detection Dataset",
          id: 28,
          desc: "",
          image: under_construction2_img,
          status: "in_progress",
          links: {
            app: "/coming_soon",
            source: "",
          },
        },
      ],
    },
  ],
};
