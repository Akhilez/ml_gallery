import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import projects from './landing/data/projects';
import ComingSoon from "./landing/coming_soon";
import 'rsuite/dist/styles/rsuite-default.css';
import MnistGanPage from "./vision/mnist_gan/mnist_gan";
import LearnLinePage from "./feed_forward/learn_line/learn_line";
import LinearClassifierPage from "./feed_forward/linear_classifier/linear_classifier";
import LearnCurvePage from './feed_forward/curve/learn_curve';
import ProfilePage from './profile/profile';
import {BrowserRouter as Router, Switch, Route} from 'react-router-dom';


export default class App extends React.Component {

    render() {
        return (
            <Router>
                <Switch>
                    <Route path="/profile" component={ProfilePage}/>
                    {
                        projects.categories.map(category => category.projects.map((project) =>
                            <Route path={project.links.app} key={project.id}>
                                {this.getProjectComponent(project)}
                            </Route>
                        ))
                    }
                    <Route path="/" component={LandingPage}/>
                </Switch>
            </Router>
        );
    }

    getProjectComponent(project) {
        switch (project.id) {
            case 1: return <LearnLinePage project={project}/>;
            case 2: return <LinearClassifierPage project={project}/>;
            case 5: return <MnistGanPage project={project}/>;
            case 9: return <LearnCurvePage project={project}/>;
            default: return <ComingSoon project={project}/>;
        }
    };

}

