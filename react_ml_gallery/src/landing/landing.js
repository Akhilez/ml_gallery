import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/components/ml_app_bar';
import MLLogo from "./ml_logo/ml_logo";
import 'cytoscape/dist/cytoscape.min';
import projectsData from './data/projects';
import Project from './components/project';
import {Centered, OutlinedButtonLink} from "../commons/components/components";


class LandingPage extends React.Component {
    render() {
        return (
            <div className={"page"}>
                <Container>
                    <MLAppBar/>
                    <MLLogo/>
                    <this.Desc/>
                    {
                        projectsData.categories.map(category => category.projects.map((project) =>
                            <Project project={project} key={project.id}/>
                        ))
                    }
                    <Footer/>
                </Container>
            </div>
        );
    }

    Desc(props) {
        return (
            <Centered>
                <div style={{fontSize: 22, marginBottom: 100}}>
                    <p>Machine Learning Gallery is a master project of few of my experiments with Neural Networks.
                        It is designed in a way to help a beginner understand the concepts with visualizations.
                        You can train and run the networks live and see the results for yourself.
                        Every project here is followed by an explanation on how it works.
                        <br/><br/>

                        Begin with a tour starting from the most basic Neural Network and build your way up.
                    </p>
                    <OutlinedButtonLink link={"/learn_line"} text={"Take a tour"}/>
                </div>
            </Centered>
        );
    }
}

function Footer() {
    return (
        <div><br/>
            <hr/>
            <br/><Centered>ML Gallery</Centered><br/></div>
    );
}

export default LandingPage;
