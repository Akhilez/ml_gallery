import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import projects from './landing/data/projects';
import ComingSoon from "./landing/coming_soon";
import 'rsuite/dist/styles/rsuite-default.css';
import MnistGanPage from "./mnist_gan/mnist_gan";
import LearnLinePage from "./regressors/learn_line";
import LinearClassifierPage from "./classifiers/linear_classifier";
import LearnCurvePage from './regressors/curve/learn_curve'


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

