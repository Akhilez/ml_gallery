import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { LinearClassifier } from "src/lib/linear/linear_classifier/LinearClassifier"

export default function () {
  return (
    <GlobalWrapper>
      <LinearClassifier />
    </GlobalWrapper>
  )
}
