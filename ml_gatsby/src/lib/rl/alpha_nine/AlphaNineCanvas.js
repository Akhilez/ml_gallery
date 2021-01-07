import React from "react"
import { Box, CircularProgress } from "@chakra-ui/core"
import { Spring } from "react-spring/renderprops"
import { mlgApi } from "src/lib/api"

const w = "w"
const b = "b"

const infoCode = {
  normal: 0,
  move_missing: 1,
  kill_position_missing: 2,
}

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
    this.posMap = this.getPosMap(this.positions)
    this.killedTargetPosition = { x: 40, y: 100 }

    // Game state
    this.gameStarted = false
    this.unused = { w: 8, b: 8 }
    this.killed = { w: 0, b: 0 }
    this.is_killing = false
    this.actionable = true
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

  handleDotClick = (e, pos) => {
    console.log(this.actionable)
    if (!this.actionable) return
    if (this.state.apiWait) return

    this.actionable = false

    const me = this.state.me
    const coord = this.posMap[pos.x + "," + pos.y].coord

    // if Phase 1
    if (this.unused[me] > -1) {
      if (this.is_killing) {
        if (pos.piece != null && pos.piece !== me) {
          this.kill(pos)
        } else {
          this.setState({
            message:
              "You selected a wrong piece. Select an opponent piece to remove",
          })
        }
        this.actionable = true
        return
      }

      this.setState({ apiWait: true })

      const [board, mens] = this.buildState()
      mlgApi.alphaNine.stepEnv(board, mens, this.state.me, coord).then(data => {
        console.log(data)

        if (data?.info?.code == null) {
          this.setState({ message: "Something went wrong, please try again" })
          this.actionable = true
          return
        }

        if (data.info.code === infoCode.normal) {
          const newIdx = this.unused[me]
          this.unused[me] -= 1

          this.state[me][newIdx].px = this.state[me][newIdx].x
          this.state[me][newIdx].py = this.state[me][newIdx].y

          this.state[me][newIdx].x = pos.x
          this.state[me][newIdx].y = pos.y

          this.state[me][newIdx].status = "alive"
          this.state[me][newIdx].coord = this.posMap[pos.x + "," + pos.y].coord
          this.state.apiWait = false
          pos.piece = me
          pos.pieceIndex = newIdx

          this.setState(this.state)
          this.swapPlayer()
        } else if (data.info.code === infoCode.kill_position_missing) {
          this.setState({ message: "Select an opponent piece to remove" })
          this.is_killing = true
        }
      })
    }
    this.actionable = true
  }

  kill(pos) {
    const idx = pos.pieceIndex
    const piece = pos.piece
    this.killed[pos.piece] += 1
    pos.piece = null
    pos.pieceIndex = null
    this.state[piece][idx].x = this.killedTargetPosition.x
    this.state[piece][idx].y = this.killedTargetPosition.y
    this.setState(this.state)
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

  buildState = () => {
    const state = []
    for (let l = 0; l < 3; l++) {
      const level = []
      for (let ce = 0; ce < 2; ce++) {
        const all4 = []
        for (let i = 0; i < 4; i++) {
          all4.push([1, 0, 0])
        }
        level.push(all4)
      }
      state.push(level)
    }

    const ws = this.state.w
    const bs = this.state.b

    for (let i = 0; i < 9; i++) {
      if (ws[i].status === "alive") {
        const idx = ws[i].coord
        state[idx[0]][idx[1]][idx[2]] = [0, 1, 0]
      }
      if (bs[i].status === "alive") {
        const idx = bs[i].coord
        state[idx[0]][idx[1]][idx[2]] = [0, 0, 1]
      }
    }

    const mens = [this.unused.w, this.unused.b, this.killed.w, this.killed.b]

    return [state, mens]
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

  getPosMap(positions) {
    const map = {}
    for (let p of positions) {
      map[p.x + "," + p.y] = p
      map[p.coord.join(",")] = p
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
