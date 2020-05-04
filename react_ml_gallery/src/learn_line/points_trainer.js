import React from "react";


export default class PointsTrainer extends React.Component {
    render() {
        return (
            <div style={{marginTop: 50, marginBottom: 50}}>
                <h2>Learn from points</h2>
                <p>Create points that are approximately linear and train the Neural Network to predict the best line equation.</p>

                <button className={"ActionButton"}>TRAIN</button>
            </div>
        );
    }
}