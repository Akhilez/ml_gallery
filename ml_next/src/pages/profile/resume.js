import React from "react";
import { Container } from "react-bootstrap";
import ProfileNavBar from "./navbar";
import { Flex, Stack, Box } from "@chakra-ui/core/dist";
import { MdFileDownload } from "react-icons/md";

export default class ResumePage extends React.Component {
  constructor(props) {
    super(props);
    this.resumeUrl = "https://storage.googleapis.com/akhilez/resume.pdf";
    // this.resumeUrl = 'https://docs.google.com/document/d/e/2PACX-1vSDPNxZzflcPBIwu0uFWfkK7S9Isa4YBIP82H4AJx-3i7UQmQYdTnkdcHrF835mmlhoMNr4EPzIo6RN/pub?embedded=true';
  }

  render() {
    return (
      <div className={"profile_root no_href"}>
        <Container>
          <ProfileNavBar active={"resume"} />
          <Stack alignItems="center">
            <Flex pb={4}>
              <h1>Resume</h1>
              <a href={this.resumeUrl} download={"latest.pdf"}>
                <Flex justifyContent="center">
                  <Box as={MdFileDownload} ml={2} mt={1} />
                </Flex>
              </a>
            </Flex>
            <embed src={this.resumeUrl} width="830px" height="1120px" />
          </Stack>
        </Container>
      </div>
    );
  }
}
