import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import {
  Link,
  Text,
  Tabs,
  TabList,
  TabPanels,
  Tab,
  TabPanel,
} from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { GridWorldCanvas } from "./GridWorldCanvas"
import { Container } from "../../components/commons"

export class GridWorld extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.grid_world
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <GridWorldCanvas />
        <Container>
          <Tabs colorScheme="brand">
            <TabList>
              <Tab>Policy Gradients</Tab>
              <Tab>Deep Q</Tab>
              <Tab>MCTS</Tab>
              <Tab>AlphaZero</Tab>
              <Tab>MuZero</Tab>
            </TabList>

            <TabPanels>
              <TabPanel>
                <p>one!</p>
              </TabPanel>
              <TabPanel>
                <p>two!</p>
              </TabPanel>
              <TabPanel>
                <p>three!</p>
              </TabPanel>
              <TabPanel>
                <p>three!</p>
              </TabPanel>
              <TabPanel>
                <p>three!</p>
              </TabPanel>
            </TabPanels>
          </Tabs>
        </Container>
      </ProjectWrapper>
    )
  }
}
