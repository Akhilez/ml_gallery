import React from "react"
import { Box, CircularProgress } from "@chakra-ui/core"
import { Spring } from "react-spring/renderprops"

const w = "w"
const b = "b"

export class AlphaNineCanvas extends React.Component {
  constructor({ parent, scale = 5, ...props }) {
    super(props)
    this.pieceRadius = 4.5
    this.state = {
      w: this.getInitialStateValues(w),
      b: this.getInitialStateValues(b),
      apiWait: false,
      me: w,
    }
    this.parent = parent
    this.scale = scale
    this.positions = this.getPositions()
    this.position_to_coords = this.getPositionsToCoords(this.positions)

    // Game state
    this.gameStarted = false
    this.unused = { w: 8, b: 8 }
    this.killed = { w: 0, b: 0 }
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
        {this.state.apiWait && (
          <CircularProgress
            isIndeterminate
            color="red.300"
            position="relative"
            top="-250px"
          />
        )}
      </Box>
    )
  }

  /*
  if in phase 1, pop form player pieces and place it on pos, build state and make api call.
  */
  handleDotClick = (e, pos) => {
    if (this.state.apiWait) return

    const me = this.state.me

    // if Phase 1
    if (this.unused[me] > -1) {
      const newIdx = this.unused[me]
      this.unused[me] -= 1

      this.state[me][newIdx].px = this.state[me][newIdx].x
      this.state[me][newIdx].py = this.state[me][newIdx].y

      this.state[me][newIdx].x = pos.x
      this.state[me][newIdx].y = pos.y

      this.setState(this.state)

      this.swapPlayer()
    }
  }

  Pieces = () => {
    const players = [b, w]
    return (
      <>
        {players.map(player =>
          this.state[player].map((piece, index) => (
            <this.Piece
              piece={piece}
              key={`${piece.x}${piece.y}`}
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
  Piece = ({ piece, index, player, ...props }) => {
    return (
      <Spring
        from={{ cx: piece.px, cy: piece.py }}
        to={{ cx: piece.x, cy: piece.y }}
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

  swapPlayer = () => {
    this.setState({ me: this.state.me === w ? b : w })
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

  getInitialStateValues(piece) {
    const values = []
    const y = piece === w ? 80 : 80 + this.pieceRadius + 2
    const leftPad = 10
    const sidePad = 2

    for (let i = 0; i < 9; i++) {
      values.push({
        x: leftPad + i * (this.pieceRadius + sidePad) + 1,
        y: y,
        px: 40,
        py: 40,
        status: "unused",
      })
    }
    return values
  }
}
