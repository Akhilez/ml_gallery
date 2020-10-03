import React from "react"
import { configure, shallow } from "enzyme"
import Adapter from "enzyme-adapter-react-16"
import { LearnCurve } from "./LearnCurve"

configure({ adapter: new Adapter() })

describe("Learn Curve", () => {
  test("order matches neural net size", () => {
    const wrapper = shallow(<LearnCurve />)
    const instance = wrapper.instance()

    const passedOrder = instance.state.order
    const weights = instance.tf.getWeights()

    console.log(weights)

    expect(passedOrder).toBe(weights.length)
  })
})
