import React from "react";
import './landing.css';
import underConstructionImage from './images/under_construction.png';
import MLAppBar from "../commons/components/ml_app_bar";
import {Centered} from "../commons/components/components";


export default function ComingSoon(props) {
    return (
        <div className={"page"}>
            <MLAppBar />
            <Centered>
            <h1>Coming Soon</h1>
            <img src={underConstructionImage} alt={"Under Construction."} />
            </Centered>
        </div>
    );
}


