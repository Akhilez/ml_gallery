import React from "react"
import { Box } from "@chakra-ui/core"

export class AlphaNineCanvas extends React.Component {
  constructor({ parent, scale = 5, ...props }) {
    super(props)
    this.parent = parent
    this.scale = scale
    this.positions = this.getPositions()
    this.position_to_coords = this.getPositionsToCoords(this.positions)
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
    return (
      <>
        {this.positions.map(p => (
          <this.Dot
            cx={p.x}
            cy={p.y}
            onClick={() => {
              console.log(p.coord)
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

  getPositions() {
    const dots = [
      [10, 70],
      [20, 60],
      [30, 50],
    ]
    const m = 40
    const positions = []
    for (let i = 0; i < 3; i++) {
      const [p, q] = dots[i]

      positions.push({ x: p, y: p, coord: [i, 0, 0] })
      positions.push({ x: q, y: p, coord: [i, 0, 1] })
      positions.push({ x: q, y: q, coord: [i, 0, 2] })
      positions.push({ x: p, y: q, coord: [i, 0, 3] })

      positions.push({ x: p, y: m, coord: [i, 1, 0] })
      positions.push({ x: m, y: p, coord: [i, 1, 1] })
      positions.push({ x: q, y: m, coord: [i, 1, 2] })
      positions.push({ x: m, y: q, coord: [i, 1, 3] })
    }
    return positions
  }

  getPositionsToCoords(positions) {
    const map = {}
    for (let p of positions) {
      map[p.coord.toString()] = p
    }
    return map
  }
}
