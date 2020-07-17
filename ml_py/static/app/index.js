import React from 'react'
import ReactDOM from 'react-dom'
import LandingPage from "./landing/landing";


window.renderHomePage = (id, props) => {
  ReactDOM.render(<LandingPage {...props} />, document.getElementById(id))
}
