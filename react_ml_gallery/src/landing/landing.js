import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/ml_app_bar';
import MLLogo from "./ml_logo/ml_logo";
import 'cytoscape/dist/cytoscape.min';
import projectsData from './data/projects';
import Project from './components/project';


class LandingPage extends React.Component {
    render() {
        return (
            <div  className={"page"}>
            <Container>
                <MLAppBar/>
                <MLLogo/>
                <Project project={projectsData.projects[0]}/>
                <Project project={projectsData.projects[1]}/>
                <Footer/>
            </Container>
            </div>
        );
    }
}

function Footer() {
    return (
        <div><br/>
            <hr/>
            <br/>Footer<br/></div>
    );
}

export default LandingPage;
