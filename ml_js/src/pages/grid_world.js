import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { GridWorld } from "src/lib/rl/grid_world/GridWorld"

export default function () {
  return (
    <GlobalWrapper>
      <GridWorld />
    </GlobalWrapper>
  )
}
