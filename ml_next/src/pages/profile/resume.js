import React from "react";
import {Container} from "react-bootstrap";
import ProfileNavBar from "./navbar";
import {Centered} from "../commons/components/components";
import {Icon} from "rsuite";
import './css/profile_style.css';


export default class ResumePage extends React.Component {
    constructor(props) {
        super(props);
        this.resumeUrl = 'https://storage.googleapis.com/akhilez/resume.pdf';
        // this.resumeUrl = 'https://docs.google.com/document/d/e/2PACX-1vSDPNxZzflcPBIwu0uFWfkK7S9Isa4YBIP82H4AJx-3i7UQmQYdTnkdcHrF835mmlhoMNr4EPzIo6RN/pub?embedded=true';
    }

    render() {
        return (
            <div className={"profile_root no_href"}>
                <Container>
                    <ProfileNavBar active={"resume"}/>
                    <Centered>
                        <h1>Resume of Akhil</h1>
                        <a href={this.resumeUrl} download={"latest.pdf"}>
                            <Icon icon="download"/> Download
                        </a>
                        <br/><br/>
                        <embed src={this.resumeUrl} width="830px" height="1120px"/>
                    </Centered>
                </Container>
            </div>
        );
    }
}
