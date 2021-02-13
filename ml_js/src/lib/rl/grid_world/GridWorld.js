import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Tabs, TabList, TabPanels, Tab, TabPanel } from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { GridWorldCanvas } from "./GridWorldCanvas"

export const GridWorld = () => (
  <ProjectWrapper project={projects.grid_world}>
    <GridWorldCanvas />
    <Tabs colorScheme="brand">
      <TabList>
        <Tab>Policy Grad</Tab>
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
  </ProjectWrapper>
)
