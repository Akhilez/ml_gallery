import React from "react"
import { configure, shallow } from "enzyme"
import Adapter from "enzyme-adapter-react-16"
import { LearnCurve } from "./LearnCurve"

configure({ adapter: new Adapter() })

let commonWrap

describe("Learn Curve", () => {
  beforeAll(() => {
    commonWrap = shallow(<LearnCurve />)
  })
  test("order matches neural net size", () => {
    const instance = commonWrap.instance()

    const passedOrder = instance.state.order
    const weights = instance.tf.getWeights()

    console.log(weights)

    expect(passedOrder).toBe(weights.length)
  })
  test("initial data shape is valid", () => {
    const instance = commonWrap.instance()
    console.log(instance.data)
  })
})
