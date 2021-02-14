import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Tabs, TabList, TabPanels, Tab, TabPanel } from "@chakra-ui/react"
import { projects } from "src/lib/globals/data"
import { GridWorldCanvas } from "./GridWorldCanvas"
import { algosList, useGridWorldStore } from "./state"

export const GridWorld = () => {
  const setAlgo = useGridWorldStore(state => state.setAlgo)
  return (
    <ProjectWrapper project={projects.grid_world}>
      <GridWorldCanvas />
      <Tabs colorScheme="brand" onChange={idx => setAlgo(algosList[idx])}>
        <TabList>
          {algosList.map(algo => (
            <Tab key={algo.id} isDisabled={algo.disabled}>
              {algo.title}
            </Tab>
          ))}
        </TabList>
        <TabPanels>
          {algosList.map(algo => (
            <TabPanel key={algo.id}>{algo.component}</TabPanel>
          ))}
        </TabPanels>
      </Tabs>
    </ProjectWrapper>
  )
}
