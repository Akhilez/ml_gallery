import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import projects from './landing/data/projects';
import ComingSoon from "./landing/coming_soon";
import 'rsuite/dist/styles/rsuite-default.css';
import MnistGanPage from "./vision/mnist_gan/mnist_gan";
import LearnLinePage from "./feed_forward/learn_line/learn_line";
import LinearClassifierPage from "./feed_forward/linear_classifier/linear_classifier";
import LearnCurvePage from './feed_forward/curve/learn_curve'


import {
    BrowserRouter as Router,
    Switch,
    Route,
} from "react-router-dom";


export default function App() {

    const getProjectComponent = function (project) {
        switch (project.id) {
            case 1: return <LearnLinePage project={project}/>;
            case 2: return <LinearClassifierPage project={project}/>;
            case 5: return <MnistGanPage project={project}/>;
            case 9: return <LearnCurvePage project={project}/>;
            default: return <ComingSoon project={project}/>;
        }
    };

    return (
        <Router>
            <Switch>
                {
                    projects.categories.map(category => category.projects.map((project) =>
                        <Route path={project.links.app} key={project.id}>
                            {getProjectComponent(project)}
                        </Route>
                    ))
                }
                <Route path="/">
                    <LandingPage/>
                </Route>
                <Route path="">
                    <LandingPage/>
                </Route>
            </Switch>
        </Router>
    );

}

