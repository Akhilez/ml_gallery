import React from "react";
import Container from "react-bootstrap/Container";
import MLAppBar from "../commons/ml_app_bar";
import '../landing/landing.css';
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import {Centered, OutlinedButtonLink} from "../commons/components/components";
import FormControl from "@material-ui/core/FormControl";
import './mnist_gan.css';
import BreadCrumb from '../commons/components/breadcrumb';
import theme from '../commons/theme';
import {MuiThemeProvider} from '@material-ui/core/styles';
import {MLHelper} from './neural_net';
import ProjectsNav from "../commons/components/projects_nav";


export default class MnistGanPage extends React.Component {

    constructor(props) {
        super(props);
        this.props = props;
        this.state = {
            selectedCharacter: "0",
        };
    }

    render(){
        new MLHelper().train(0);
        return (
            <div className={"page"}>
                <div style={{float: "left"}}>
                    <ProjectsNav activeKey={this.props.project.id}/>
                </div>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.path}/>
                    <Centered>
                        <h1>MNIST GAN</h1>
                        <p className={"header_desc"}>Generate AI-powered hand-drawn character images.</p>

                        <OutlinedButtonLink text={"How it works"} link={"#how_it_works"}/><br/>

                        <MuiThemeProvider theme={theme()}>
                            <FormControl variant="outlined" id={"charDropdown"}>
                                <InputLabel id="character-selector">Character</InputLabel>
                                <Select variant={"outlined"}
                                        labelId="character-selector"
                                        id="demo-simple-select-outlined"
                                        value={this.state.selectedCharacter}
                                        onChange={(event) => {
                                            this.setSelectedCharacter(event.target.value)
                                        }}
                                        label="Character"
                                >
                                    {[0, 1, 2, 3, 4, 5, 6, 7, 8, 9].map((index, value) =>
                                        <MenuItem value={value} key={index}>{value}</MenuItem>
                                    )}
                                </Select>
                            </FormControl>
                        </MuiThemeProvider>
                        <button className={"ActionButton"}>GENERATE</button>

                    </Centered>
                </Container>
            </div>
        )
    }

    setSelectedCharacter(character) {
        this.setState((state, props) => {
            return {selectedCharacter: character}
        });
    }
}