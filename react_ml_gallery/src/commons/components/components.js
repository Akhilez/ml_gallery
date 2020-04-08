import React from "react";

export function Centered (props) {
    return (
        <div align={"center"}>
            {props.children}
        </div>
    );
}

export function OutlinedButtonLink(props) {
    return (
        <a href={props.link}>{props.text}</a>
    );
}
