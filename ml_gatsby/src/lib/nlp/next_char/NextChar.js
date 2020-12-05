import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Box, Input, Text } from "@chakra-ui/core"
import { projects } from "src/lib/globals/data"

export class NextChar extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      text: "",
      predicted: "",
    }
    this.project = projects.next_char
  }
  handleInput(event) {
    console.log(this.state)
    this.setState({ text: event.target.value, predicted: "ding dong" })
    // TODO: call the freaking API and set prediction
  }
  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <Box width="4xl" height="120px" mt={4}>
            <Input
              size="lg"
              height="100px"
              fontSize="5xl"
              focusBorderColor="red.400"
              backgroundColor="transparent"
              onChange={event => this.handleInput(event)}
            />
            <Text
              fontSize="5xl"
              color="gray.300"
              mt="-85px"
              textAlign="left"
              ml={4}
            >
              {`${this.state.text}${this.state.predicted}`}
            </Text>
          </Box>
        </Centered>
      </ProjectWrapper>
    )
  }
}
