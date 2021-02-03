import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { LearnCurve } from "src/lib/linear/learn_curve/LearnCurve"

export default function () {
  return (
    <GlobalWrapper>
      <LearnCurve />
    </GlobalWrapper>
  )
}
