import React from "react";
import {Container} from "react-bootstrap";
import ProfileNavBar from "./navbar";
import {ProfileBadge, GithubCalendar} from './profile_components';
import toggle_img from './media/toggle.png';
import './css/timeline.css';
import code_art_img from './media/cover_code_art_with_bg_dark.png';


export default class ProfilePage extends React.Component {
    render() {
        return (
            <div className={"profile_root"}>
                <Container>
                    <ProfileNavBar active={"profile"}/>
                    <ProfileBadge/>
                    <this.Bio/>
                    <this.TimeLine/>
                    <this.CodingActivity/>
                </Container>
            </div>
        );
    }

    Bio(props) {
        return (
            <div>
                <h3 className="header1">Bio</h3>
                <p className="para no_href_p">I am a Master’s student at <a href="https://www.uc.edu/">University of
                    Cincinnati</a> majoring in <a
                    href="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/detail/20MAS-AI-MENG">Artificial
                    Intelligence</a>, specializing in Deep Learning architectures for Computer Vision, Reinforcement
                    Learning
                    and Complex Intelligent Systems. Previously, I was a full-stack engineer at an AI based startup
                    called <a href="https://www.aviso.com/">Aviso</a>, where I took the ownership of an internal web-app
                    that
                    manages
                    the cloud infrastructure. During my undergrad, I worked as a part-time Software Developer at the
                    college’s
                    administrative department where I developed software applications for digitalization and automation
                    of
                    the
                    administrative operations.</p>
                <p className="para no_href_p">In my spare time, I work on my own <a
                    href="http://akhilez.com/home/resume/">independent
                    projects</a>. I
                    developed a number of applications for the web and mobile over the years because I enjoy coding and
                    designing.
                    Lately, I’ve developed a deep interest in Artificial Intelligence and Space. Now, I associate my
                    goals
                    strongly
                    with pioneering the advancements in Artificial General Intelligence for further space exploration
                    and
                    more.</p>

                <div className="header1" style={{fontSize: 20, paddingTop: 20}}>
                    <img src={toggle_img} alt="toggle" height="30px"/>
                    &nbsp; Available for hire.
                </div>
            </div>
        );
    }

    TimeLine(props) {
        return (
            <div>
                <h3 className="header1">Timeline</h3>

                <div id="timeline_section">

                    <ul className="timeline no_href">
                        <li className="event" data-date="2019">
                            <h3><a
                                href="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/detail/20MAS-AI-MENG">Master’s
                                in
                                Artificial Intelligence</a></h3>
                            <p><a href="https://www.uc.edu/">University of Cincinnati</a></p>
                            <p><a
                                href="https://webapps2.uc.edu/ecurriculum/degreeprograms/program/majormap/20MAS-AI-MENG">Courses
                                taken:</a> Intelligent Systems, ML, AI, Deep Learning, Complex Systems, Computer Vision,
                                StartupUC</p>
                        </li>
                        <li className="event" data-date="2018">
                            <h3>Full-Stack Engineer</h3>
                            <p><a href="https://www.aviso.com/">Aviso Inc.</a></p>
                            <p>Worked on a wide variety of tasks revolving around the cloud infrastructure of the Aviso
                                AI
                                product.</p>
                        </li>
                        <li className="event" data-date="2015">
                            <h3>Part-Time Developer</h3>
                            <p><a href="https://kmit.in/home">Keshav Memorial Institute of Technology</a></p>
                            <p>Developed apps for the college’s operations like <a
                                href="http://akhilez.com/home/all_projects//#student_feedback">Student
                                Feedback</a> and <a
                                href="http://akhilez.com/home/all_projects//#gatepass">Gate-Pass System</a></p>
                        </li>
                        <li className="event" data-date="2014">
                            <h3>Bachelor's in Computer Science and Engineering</h3>
                            <p><a href="https://kmit.in/home">Keshav Memorial Institute of Technology</a></p>
                        </li>
                    </ul>
                </div>
            </div>
        );
    }

    CodingActivity(props) {
        return (
            <div>
                <h3 className="header1 no_href_p"><a target="_blank" href="https://github.com/Akhilez" style={{fontWeight: 700}}>
                    Coding Activity</a></h3>

                <GithubCalendar/>

                <div className="row">
                    <img src={code_art_img} alt="CoverPhoto" width="400"/>
                </div>
            </div>
        );
    }

}
