import React from "react";
import {Row, Col} from 'react-bootstrap';
import profile_photo from './media/profile_photo.jpg';
import './css/profile_style.css';
import './css/fontawesome/css/font-awesome.min.css';
import './css/fontawesome/css/fonts.css';
import ReactGithubCalendar from '@axetroy/react-github-calendar';

export function ProfileBadge(props) {
    return (
        <div className={"profileBadgeContainer"}>
            <Row>
                <Col sm={"auto"}>
                    <img src={profile_photo} className={"profile_img"} alt={"profilePhoto"}/>
                </Col>
                <Col sm={"auto"}>
                    <div style={{marginTop: 30}}>
                        <h1 className={"light_font"} style={{fontSize: 58}}>Akhil Devarashetti</h1>

                        <div style={{fontSize: 28, marginTop: -10}}>
                            Computer Science Engineer
                        </div>
                        <div style={{fontSize: 18}} className="roboto-light-ak no_href"><a
                            href="mailto:akhilkannadev@gmail.com">akhilkannadev@gmail.com</a></div>

                        <Social/>

                        <div>
                            <a target="_blank" href="http://akhilez.com/home/resume/"
                               className="btn btn-outline-secondary resume-button">RESUME</a>
                        </div>
                    </div>
                </Col>
            </Row>
        </div>
    );
}

export function Social(props) {
    return (
        <div className="social">
            <ul>
                <li><a target="_blank" href="https://www.linkedin.com/in/akhilez/">
                    <i className="fa fa-linkedin"/></a></li>
                <li><a target="_blank" href="https://github.com/Akhilez">
                    <i className="fa fa-github"/></a>
                </li>
            </ul>
        </div>
    );
}

export class GithubCalendar extends React.Component {
    render() {
        return (
            <div>
                <ReactGithubCalendar name="axetroy" />
            </div>
        );
    }
}
