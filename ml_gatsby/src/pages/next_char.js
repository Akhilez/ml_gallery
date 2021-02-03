import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { NextChar } from "src/lib/nlp/next_char/NextChar"

export default function () {
  return (
    <GlobalWrapper>
      <NextChar />
    </GlobalWrapper>
  )
}
