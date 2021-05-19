import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { LandingPage } from "src/lib/landing/LandingPage"

export default function Home() {
  return (
    <GlobalWrapper>
      <LandingPage />
    </GlobalWrapper>
  )
}
