import React from 'react';
import logo from './logo.svg';
import './App.css';
import LandingPage from "./landing/landing";
import MnistGanPage from "./mnist_gan";

import {
  BrowserRouter as Router,
  Switch,
  Route,
  Link
} from "react-router-dom";


export default function App() {
  return (
      <Router>
          {/* A <Switch> looks through its children <Route>s and
            renders the first one that matches the current URL. */}
          <Switch>
            <Route path="/mnist_gan">
              <MnistGanPage />
            </Route>
            <Route path="/">
              <LandingPage />
            </Route>
          </Switch>
      </Router>
  );
}

