import React from 'react';
import Container from "@material-ui/core/Container";
import MLAppBar from '../commons/ml_app_bar';
import MLLogo from "./ml_logo/ml_logo";
import 'cytoscape/dist/cytoscape.min';


class LandingPage extends React.Component {
    render() {
        return (
            <Container>
                <MLAppBar/>
                <MLLogo/>
                <Project/>
            </Container>
        );
    }
}

class Project extends React.Component {
    render() {
        return (
            <div>
                Project
            </div>
        );
    }
}

export default LandingPage;
