import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Box, Flex, Input, Text } from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { mlgApi } from "src/lib/api"

export class NextChar extends React.Component {
  constructor(props) {
    super(props)
    this.state = {
      text: "",
      predicted: "... Enter text here!",
      history: [],
    }
    this.project = projects.next_char
    this.history_max_len = 10
  }

  handleInput(event) {
    const text = event.target.value
    this.setState({ text })
    if (!text) this.setState({ predicted: "... Enter text here!" })
    else
      mlgApi
        .nextChar(text)
        .then(res => res.json())
        .then(result => {
          this.setState({ predicted: result.pred })
          this.enqueue(text, result.pred)
        })
    // TODO: Dude, if entered text goes far, then set negative margin to the placeholder or something
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <Box
            width={{ base: "sm", md: "2xl", lg: "4xl" }}
            height={{ base: "75px", md: "120px" }}
            mt={4}
          >
            <Input
              size="lg"
              height={{ base: "70px", md: "100px" }}
              fontSize={{ base: "xl", md: "5xl" }}
              focusBorderColor="red.400"
              backgroundColor="transparent"
              onChange={event => this.handleInput(event)}
              borderRadius={{ md: "10px" }}
            />
            <Text
              fontSize={{ base: "xl", md: "5xl" }}
              color="gray.300"
              mt={{ base: "-50px", md: "-85px" }}
              textAlign="left"
              ml={4}
              whiteSpace="nowrap"
              overflow="hidden"
            >
              {`${this.state.text}${this.state.predicted}`}
            </Text>
          </Box>
          <Box width="4xl" overflowY="auto" maxH="150px" mb={4}>
            {this.state.history.reverse().map(history => (
              <Flex whiteSpace="pre">
                <Text>{history.text}</Text>
                <Text color="gray.400">{`${history.predicted}`}</Text>
              </Flex>
            ))}
          </Box>
          <Text>
            Firstly, TYPE SLOW! The Neural Network is running on a tiny CPU many
            miles away!
            <br />
            <br />
            The fun part is that this model is trained on the subtitles of 7
            Marvel movies! 🤩
            <br />
            Try typing in some words or dialogs from Avengers and it predict the
            words character-by-character that are somewhat related.
            <br />
            <br />
            <b>Here's a fun challenge</b> - Guess the 7 marvel movies used in
            this model. 😉
            <br />
            Try to type in a dialog specific to a movie in mind, if the
            predicted text doesn't make sense, it was probably not used.
          </Text>
        </Centered>
      </ProjectWrapper>
    )
  }

  enqueue = async (text, predicted) => {
    const hist = this.state.history
    if (hist.length >= this.history_max_len) {
      hist.shift()
    }
    hist.push({ text, predicted })
    this.setState({ history: hist })
  }
}
