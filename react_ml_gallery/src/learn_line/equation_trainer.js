import {FlexboxGrid, Input} from 'rsuite';
import React from "react";
import './learn_line.css';
import '../commons/components/components.css';


export default class EquationTrainer extends React.Component {
    render() {
        return (
            <div style={{marginTop: 50, marginBottom: 50}}>
                <h3>Learn from equation</h3>
                <p>Set "m" and "c" values and train the Neural Network to predict these values.</p>
                <div style={{fontSize: 40}}>
                    <FlexboxGrid>
                        <FlexboxGrid.Item>y = </FlexboxGrid.Item>
                        <FlexboxGrid.Item>
                            <this.ParamsPicker params={"M"}/>
                        </FlexboxGrid.Item>
                        <FlexboxGrid.Item> x + </FlexboxGrid.Item>
                        <FlexboxGrid.Item>
                            <this.ParamsPicker params={"C"}/>
                        </FlexboxGrid.Item>
                    </FlexboxGrid>
                </div>
                <button className={"ActionButton"}>TRAIN</button>
            </div>
        );
    }

    ParamsPicker(props) {
        if (props.params === "M")
            return (
                <Input className={"inputBox"} placeholder="m" type={"number"} />
            );
        else if (props.params === "C")
            return (
                <Input className={"inputBox"} placeholder="c" type={"number"} />
            );
    }
}