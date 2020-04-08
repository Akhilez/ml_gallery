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
import {Switch, Route, useRouteMatch} from "react-router-dom";
import MnistGanReport from "./report";
import BreadCrumb from '../commons/components/breadcrumb';


export default function MnistGanPage() {
    let match = useRouteMatch();
    return (
        <Switch>
            <Route path={`${match.path}/report`}><MnistGanReport/></Route>
            <Route path={match.path}> <MnistGanMain/> </Route>
        </Switch>
    );
}

class MnistGanMain extends React.Component {
    constructor(props) {
        super(props);
        this.props = props;
        this.state = {
            selectedCharacter: "0",
        };
        this.path = '/mnist_gan';
    }

    render() {
        return (
            <div className={"page"}>
                <Container>
                    <MLAppBar/>
                    <BreadCrumb path={this.path}/>
                    <Centered>
                        <h1>MNIST GAN</h1>
                        <p className={"header_desc"}>Generate AI-powered hand-drawn character images.</p>

                        <OutlinedButtonLink text={"How it works"} link={"/mnist_gan/report"}/><br/>

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
                                    <MenuItem value={value}>{value}</MenuItem>
                                )}
                            </Select>
                        </FormControl>
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