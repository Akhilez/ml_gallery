import React from "react";
import './components.css';

export function Centered(props) {
    return (
        <div align={"center"}>
            {props.children}
        </div>
    );
}

export function OutlinedButtonLink(props) {
    return (
        <div>
            <a className={"outlinedButtonLink"} href={props.link}>{props.text}</a>
        </div>
    );
}


export function SizedBox(props) {
    return (
        <div style={{width: props.width, height: props.height}}/>
    )
}