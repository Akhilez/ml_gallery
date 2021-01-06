import React from "react"
import { Box } from "@chakra-ui/core"
import { Spring } from "react-spring/renderprops"

const w = "w"
const b = "b"

export class AlphaNineCanvas extends React.Component {
  constructor({ parent, scale = 5, ...props }) {
    super(props)
    this.pieceRadius = 4.5
    this.state = {
      pos: {
        w: this.getInitialUnusedPositions(w),
        b: this.getInitialUnusedPositions(b),
      },
      pos_prev: {
        w: this.getInitialPrevPositions(),
        b: this.getInitialPrevPositions(),
      },
    }
    this.parent = parent
    this.scale = scale
    this.positions = this.getPositions()
    this.position_to_coords = this.getPositionsToCoords(this.positions)

    // Game state
    this.gameStarted = false
    this.currentPlayer = w
    this.pieceStack = { w: 8, b: 8 }
  }

  render() {
    const realSide = this.scale * 60
    return (
      <Box w={`${realSide}px`} h={`${realSide}px`}>
        <svg viewBox={`0 0 80 100`}>
          <this.LineFrame />
          <this.PositionalDots />
          <this.Pieces />
        </svg>
      </Box>
    )
  }

  Pieces = () => {
    const players = [b, w]
    return (
      <>
        {players.map(player =>
          this.state.pos[player].map((pos, index) => (
            <this.Piece
              pos={pos}
              key={pos}
              player={player}
              fill={player === b ? "black" : "white"}
              index={index}
            />
          ))
        )}
      </>
    )
  }

  PositionalDots = () => {
    return (
      <>
        {this.positions.map(p => (
          <this.Dot
            cx={p.x}
            cy={p.y}
            onClick={e => this.handleDotClick(e, p)}
            key={p.pos}
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
  Dot = props => (
    <circle
      {...props}
      r={1.5}
      stroke="none"
      fill="red"
      onMouseOver={event => event.target.setAttribute("r", "2")}
      onMouseOut={e => e.target.setAttribute("r", "1.5")}
    />
  )
  Piece = ({ pos, index, player, ...props }) => {
    const prev_pos = this.state.pos_prev[player][index]
    return (
      <Spring
        from={{ cx: prev_pos[0], cy: prev_pos[1] }}
        to={{ cx: pos[0], cy: pos[1] }}
      >
        {springProps => (
          <circle
            {...props}
            cx={springProps.cx}
            cy={springProps.cy}
            r={2}
            strokeWidth={0.5}
            stroke="black"
          />
        )}
      </Spring>
    )
  }

  /*
  if in phase 1, pop form player pieces and place it on pos.
  */
  handleDotClick = (e, pos) => {
    // if Phase 1
    if (this.pieceStack[this.currentPlayer] > -1) {
      const menIdx = this.pieceStack[this.currentPlayer]
      this.pieceStack[this.currentPlayer] -= 1

      this.state.pos_prev[this.currentPlayer][menIdx] = this.state.pos[
        this.currentPlayer
      ][menIdx]
      this.state.pos[this.currentPlayer][menIdx] = [pos.x, pos.y]

      this.setState({ pos: this.state.pos, pos_prev: this.state.pos_prev })

      this.swapPlayer()
    }
  }

  swapPlayer = () => {
    this.currentPlayer = this.currentPlayer === w ? b : w
  }

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

  getInitialUnusedPositions(piece) {
    const pos = []
    const y = piece === "w" ? 80 : 80 + this.pieceRadius + 2
    const leftPad = 10
    const sidePad = 2
    for (let i = 0; i < 9; i++)
      pos.push([leftPad + i * (this.pieceRadius + sidePad) + 1, y])
    console.log(pos)
    return pos
  }

  getInitialPrevPositions() {
    const pos = []
    for (let i of this.range(0, 9)) pos.push([40, 40])
    return pos
  }

  range(start, end) {
    return Array.from({ length: end - start + 1 }, (_, i) => i)
  }
}
