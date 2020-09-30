import React from "react"
import { LearnCurveTF } from "./tf_code"
import { LearnCurve } from "./LearnCurve"

test("order matches neural net size", () => {
  const component = <LearnCurve />
  const learnCurve = new LearnCurveTF(component)
  console.log(learnCurve.getWeights())
})
