import React from "react";
import {Row, Col} from 'react-bootstrap';
import profile_photo from './media/profile_photo.jpg';
import './css/profile_style.css';
import './css/fontawesome/css/font-awesome.min.css';
import './css/fontawesome/css/fonts.css';
import './css/github_calendar.css';

export function ProfileBadge(props) {
    return (
        <div className={"profileBadgeContainer"}>
            <Row>
                <Col sm={"auto"}>
                    <img src={profile_photo} className={"profile_img"} alt={"profilePhoto"}/>
                </Col>
                <Col sm={"auto"}>
                    <div style={{marginTop: 20}}>
                        <h1 className={"light_font"} style={{fontSize: 58}}>Akhil D. <i style={{fontSize: 35}}>(Akhilez)</i></h1>

                        <div style={{fontSize: 28, marginTop: -10}}>
                            Machine Learning Engineer
                        </div>
                        <div style={{fontSize: 22}} className="roboto-light-ak no_href"><a
                            href="mailto: ak@akhil.ai"> ak@akhil.ai</a></div>

                        <Social/>
                        <ResumeButton/>
                    </div>
                </Col>
            </Row>
        </div>
    );
}

export function ResumeButton(props) {
    return (
        <div>
            <a target="_blank" rel="noopener noreferrer" href="http://akhilez.com/home/resume/"
               className="btn btn-outline-secondary resume-button">RESUME</a>
        </div>
    );
}

export function Social(props) {
    return (
        <div className="social">
            <ul>
                <li><a target="_blank" rel="noopener noreferrer" href="https://www.linkedin.com/in/akhilez/">
                    <i className="fa fa-linkedin"/></a></li>
                <li><a target="_blank" rel="noopener noreferrer" href="https://github.com/Akhilez">
                    <i className="fa fa-github"/></a>
                </li>
            </ul>
        </div>
    );
}

export class GithubCalendar extends React.Component {
    render() {
        return (
            <div className="calendar" style={{marginTop: 20, marginBottom: 50, width: "100%"}}>
                Loading the data just for you.
            </div>
        );
    }

    componentDidMount() {
        window.GitHubCalendar(".calendar", "Akhilez", {responsive: true});
    }
}
