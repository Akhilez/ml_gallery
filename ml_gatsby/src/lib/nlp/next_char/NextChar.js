import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Box, Input, Text } from "@chakra-ui/core"
import { projects } from "src/lib/globals/data"
import { mlgApi } from "src/lib/api"

export class NextChar extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      text: "",
      predicted: "... Enter text here!",
    }
    this.project = projects.next_char
  }
  handleInput(event) {
    const text = event.target.value
    this.setState({ text })
    if (!text) this.setState({ predicted: "... Enter text here!" })
    else
      mlgApi
        .nextChar(text)
        .then(res => res.json())
        .then(result => this.setState({ predicted: result.pred }))
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
              pr="200px"
              onChange={event => this.handleInput(event)}
            />
            <Text
              fontSize="5xl"
              color="gray.300"
              mt="-85px"
              textAlign="left"
              ml={4}
            >
              {this.state.predicted}
            </Text>
          </Box>
        </Centered>
      </ProjectWrapper>
    )
  }
}
