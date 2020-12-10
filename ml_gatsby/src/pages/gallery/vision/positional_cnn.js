import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { PositionalCNN } from "src/lib/vision/positional_cnn/PositionalCNN"

export default function () {
  return (
    <GlobalWrapper>
      <PositionalCNN />
    </GlobalWrapper>
  )
}
