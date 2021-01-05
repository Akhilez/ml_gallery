import React from "react"
import { Box } from "@chakra-ui/core"

export class AlphaNineCanvas extends React.Component {
  constructor({ parent, scale = 5, ...props }) {
    super(props)
    this.parent = parent
    this.scale = scale
  }

  render() {
    const realSide = this.scale * 60
    return (
      <Box w={`${realSide}px`} h={`${realSide}px`}>
        <svg viewBox={`0 0 80 80`}>
          <this.LineFrame />
          <this.PositionalDots />
        </svg>
      </Box>
    )
  }

  PositionalDots = () => {
    const dots = [
      [10, 70],
      [20, 60],
      [30, 50],
    ]
    const m = 40
    const positions = []
    for (let i = 0; i < 3; i++) {
      const [p, q] = dots[i]

      positions.push([p, p, [i, 0, 0]])
      positions.push([q, p, [i, 0, 1]])
      positions.push([q, q, [i, 0, 2]])
      positions.push([p, q, [i, 0, 3]])

      positions.push([p, m, [i, 1, 0]])
      positions.push([m, p, [i, 1, 1]])
      positions.push([q, m, [i, 1, 2]])
      positions.push([m, q, [i, 1, 3]])
    }
    return (
      <>
        {positions.map(p => (
          <this.Dot
            cx={p[0]}
            cy={p[1]}
            onClick={() => {
              console.log(p[2])
            }}
          />
        ))}
      </>
    )
  }

  LineFrame = () => {
    const Line = this.Line
    const [a1, a2] = [10, 70]
    const [b1, b2] = [20, 60]
    const [c1, c2] = [30, 50]
    const m = 40
    return (
      <>
        <Line x1={a1} y1={a1} x2={a2} y2={a1} />
        <Line x1={a2} y1={a1} x2={a2} y2={a2} />
        <Line x1={a2} y1={a2} x2={a1} y2={a2} />
        <Line x1={a1} y1={a2} x2={a1} y2={a1} />

        <Line x1={b1} y1={b1} x2={b2} y2={b1} />
        <Line x1={b2} y1={b1} x2={b2} y2={b2} />
        <Line x1={b2} y1={b2} x2={b1} y2={b2} />
        <Line x1={b1} y1={b2} x2={b1} y2={b1} />

        <Line x1={c1} y1={c1} x2={c2} y2={c1} />
        <Line x1={c2} y1={c1} x2={c2} y2={c2} />
        <Line x1={c2} y1={c2} x2={c1} y2={c2} />
        <Line x1={c1} y1={c2} x2={c1} y2={c1} />

        <Line x1={a1} y1={m} x2={c1} y2={m} />
        <Line x1={c2} y1={m} x2={a2} y2={m} />
        <Line x1={m} y1={a1} x2={m} y2={c1} />
        <Line x1={m} y1={c2} x2={m} y2={a2} />
      </>
    )
  }

  Line = props => <line {...props} stroke="red" strokeWidth="1" />
  Dot = props => <circle {...props} r={1.5} stroke="none" />
}
