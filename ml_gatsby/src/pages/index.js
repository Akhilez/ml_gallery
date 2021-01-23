import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import LandingPage from "src/lib/landing/LandingPage"
import { LandingPageV2 } from "../lib/landing/LandingPageV2"

export default function Home() {
  return (
    <GlobalWrapper>
      <LandingPageV2 />
    </GlobalWrapper>
  )
}
