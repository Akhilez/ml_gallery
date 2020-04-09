import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import projects from './landing/data/projects';
import ComingSoon from "./landing/coming_soon";

import {
    BrowserRouter as Router,
    Switch,
    Route,
} from "react-router-dom";
import MnistGanPage from "./mnist_gan/mnist_gan";
import LearnLinePage from "./learn_line/learn_line";


export default function App() {
    let routerTargets = {
        1: <LearnLinePage />,
        2: <ComingSoon />,
        3: <ComingSoon />,
        4: <ComingSoon />,
        5: <MnistGanPage />,
        6: <ComingSoon />,
        7: <ComingSoon />,
        8: <ComingSoon />,
    };
    return (
        <Router>
            <Switch>
                {
                    projects.projects.map((project) => {
                        return (
                            <Route path={project.links.app} key={project.id}>
                                {routerTargets[project.id]}
                            </Route>
                        );
                    })
                }
                <Route path="/">
                    <LandingPage/>
                </Route>
            </Switch>
        </Router>
    );

}

