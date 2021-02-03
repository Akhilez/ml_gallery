import React from "react"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"
import { MnistGan } from "src/lib/vision/mnist_gan/MnistGan"

export default function () {
  return (
    <GlobalWrapper>
      <MnistGan />
    </GlobalWrapper>
  )
}
