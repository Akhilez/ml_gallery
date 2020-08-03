import React from "react";
import logo from "../../../../ml_next/public/media/ml_logo.png";

class MLLogo extends React.Component {
  render() {
    let imgStyle = {
      marginTop: "100px",
      marginBottom: "100px",
      maxWidth: "100%",
      height: "auto",
    };
    return (
      <div align="center">
        <img alt="ml_logo" src={logo} style={imgStyle} />
      </div>
    );
  }
}

export default MLLogo;
