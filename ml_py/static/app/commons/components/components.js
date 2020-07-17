import React from "react";
import './components.css';
import {Helmet} from 'react-helmet';

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


const TitleComponent = ({ title }) => {
    let defaultTitle = 'ML Gallery • Akhilez';
    return (
        <Helmet>
            <title>{title ? title : defaultTitle}</title>
        </Helmet>
    );
};
export { TitleComponent };
