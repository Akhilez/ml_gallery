import React from "react";
import "./App.css";
import LandingPage from "./landing/landing";
import projects from "./landing/data/projects";
import ComingSoon from "./landing/coming_soon";
import "rsuite/dist/styles/rsuite-default.css";
import MnistGanPage from "./vision/mnist_gan/mnist_gan";
import LearnLinePage from "./feed_forward/learn_line/learn_line";
import LinearClassifierPage from "./feed_forward/linear_classifier/linear_classifier";
import LearnCurvePage from "./feed_forward/curve/learn_curve";
import ProfilePage from "./profile/profile";
import { TitleComponent } from "./commons/components/components";
import ResumePage from "./profile/resume";
import urls from "./urls";
import AllProjectsPage from "./profile/all_projects";
import DeepIrisPage from "./feed_forward/deep_iris/deep_iris";
import WhichCharPage from "./vision/which_char/which_char";
import NumberDetectorPage from "./vision/num_detector/num_detector";
import { ThemeProvider } from "@chakra-ui/core";
import { Router } from "@reach/router";

export default class App extends React.Component {
  render() {
    // tf.setBackend('wasm');
    return (
      <>
        <Router></Router>
        <Router>
          <Switch>
            {projects.categories.map((category) =>
              category.projects.map((project) => (
                <Route path={project.links.app} key={project.id}>
                  {this.getProjectComponent(project)}
                </Route>
              ))
            )}
            <Route
              path={urls.resume.url}
              component={this.withTitle({
                component: ResumePage,
                title: urls.resume.title,
              })}
            />
            <Route path={urls.ml_gallery.url} component={LandingPage} />
            <Route
              path={urls.all_projects.url}
              component={this.withTitle({
                component: AllProjectsPage,
                title: urls.all_projects.title,
              })}
            />

            <Route
              path={urls.profile.url}
              component={this.withTitle({
                component: ProfilePage,
                title: urls.profile.title,
              })}
            />
          </Switch>
        </Router>
      </>
    );
  }

  getProjectComponent(project) {
    switch (project.id) {
      case 1:
        return <LearnLinePage project={project} />;
      case 2:
        return <LinearClassifierPage project={project} />;
      case 3:
        return <DeepIrisPage project={project} />;
      case 4:
        return <WhichCharPage project={project} />;
      case 5:
        return <MnistGanPage project={project} />;
      case 7:
        return <NumberDetectorPage project={project} />;
      case 9:
        return <LearnCurvePage project={project} />;
      default:
        return <ComingSoon project={project} />;
    }
  }

  withTitle({ component: Component, title }) {
    return class Title extends Component {
      render() {
        return (
          <React.Fragment>
            <TitleComponent title={title} />
            <Component {...this.props} />
          </React.Fragment>
        );
      }
    };
  }
}
