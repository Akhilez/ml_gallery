import React from "react";
import Container from "react-bootstrap/Container";
import MLAppBar from "../../commons/components/ml_app_bar";
import '../../landing/landing.css';
import InputLabel from "@material-ui/core/InputLabel";
import Select from "@material-ui/core/Select";
import MenuItem from "@material-ui/core/MenuItem";
import {Centered, OutlinedButtonLink} from "../../commons/components/components";
import FormControl from "@material-ui/core/FormControl";
import './mnist_gan.css';
import BreadCrumb from '../../commons/components/breadcrumb';
import {createMuiTheme, MuiThemeProvider} from '@material-ui/core/styles';
import ProjectsNav from "../../commons/components/projects_nav";


function theme() {
    return createMuiTheme({
        palette: {
            primary: {
                // light: will be calculated from palette.primary.main,
                main: '#f44336',
                // dark: will be calculated from palette.primary.main,
                // contrastText: will be calculated to contrast with palette.primary.main
            },
            secondary: {
                light: '#0066ff',
                main: '#e91e63',
                // dark: will be calculated from palette.secondary.main,
                contrastText: '#ffffff',
            },
            // Used by `getContrastText()` to maximize the contrast between
            // the background and the text.
            contrastThreshold: 3,
            // Used by the functions below to shift a color's luminance by approximately
            // two indexes within its tonal palette.
            // E.g., shift from Red 500 to Red 300 or Red 700.
            tonalOffset: 0.2,
        },
    });
}

export default class MnistGanPage extends React.Component {

    constructor(props) {
        super(props);
        this.props = props;
        this.state = {
            selectedCharacter: "0",
        };
    }

    render(){
        // (new MLHelper()).train(0);
        return (
            <div className={"page"}>
                <div style={{float: "left"}}>
                    <ProjectsNav activeKey={this.props.project.id}/>
                </div>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.props.project.links.app}/>
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