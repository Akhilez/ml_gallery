import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { FindAllChars } from "src/lib/vision/find_all_chars/FindAllChars"

export default function () {
  return (
    <GlobalWrapper>
      <FindAllChars />
    </GlobalWrapper>
  )
}
