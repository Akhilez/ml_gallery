import React from 'react'
import ReactDOM from 'react-dom'


function Welcome(props) {
  console.log(window.context);
  return <h1>Hello!</h1>;
}

ReactDOM.render(
  <Welcome name="world" />,
  document.getElementById('react')
);