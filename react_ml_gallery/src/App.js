import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import projects from './landing/data/projects';
import ComingSoon from "./landing/coming_soon";
import 'rsuite/dist/styles/rsuite-default.css';


import {
    BrowserRouter as Router,
    Switch,
    Route,
} from "react-router-dom";
import MnistGanPage from "./mnist_gan/mnist_gan";
import LearnLinePage from "./learn_line/learn_line";


export default function App() {

    const getProjectComponent = function (project) {
        let routerTargets = {
            1: <LearnLinePage project={project}/>,
            2: <ComingSoon project={project}/>,
            3: <ComingSoon project={project}/>,
            4: <ComingSoon project={project}/>,
            5: <MnistGanPage project={project}/>,
            6: <ComingSoon project={project}/>,
            7: <ComingSoon project={project}/>,
            8: <ComingSoon project={project}/>,
        };
        return routerTargets[project.id];
    };

    return (
        <Router>
            <Switch>
                {
                    projects.projects.map((project) => {
                        return (
                            <Route path={project.links.app} key={project.id}>
                                {getProjectComponent(project)}
                            </Route>
                        );
                    })
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

