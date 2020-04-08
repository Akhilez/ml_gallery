import React from 'react';
import './App.css';
import LandingPage from "./landing/landing";
import MnistGanPage from "./mnist_gan/mnist_gan.js";

import {
  BrowserRouter as Router,
  Switch,
  Route,
} from "react-router-dom";


export default function App() {
  return (
      <Router>
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

