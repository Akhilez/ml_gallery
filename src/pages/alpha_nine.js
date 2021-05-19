import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { AlphaNine } from "src/lib/rl/alpha_nine/AlphaNine"

export default function () {
  return (
    <GlobalWrapper>
      <AlphaNine />
    </GlobalWrapper>
  )
}
