import React from "react"
import { Box, CircularProgress } from "@chakra-ui/react"
import { Spring } from "react-spring/renderprops"
import { mlgApi } from "src/lib/api"

const w = "w"
const b = "b"

const infoCode = {
  normal: 0,
  bad_action_position: 1,
  bad_move: 2,
  bad_kill_position: 3,
}

export class AlphaNineCanvas extends React.Component {
  constructor({ parent, scale = 5, ...props }) {
    super(props)

    // UI related
    this.pieceRadius = 4.5
    this.scale = scale
    this.killedTargetPosition = { x: 40, y: -10 }

    this.state = {
      w: this.getInitialStateValues(w),
      b: this.getInitialStateValues(b),
      apiWait: false,
      me: w,
      scores: { w: 0, b: 0 },
    }
    this.parent = parent

    // Game state
    this.positions = this.getPositions()
    this.posMap = this.getPosMap(this.positions)
    this.unused = { w: 8, b: 8 }
    this.killed = { w: 0, b: 0 }
    this.actionable = true
    this.currentAction = {}
  }

  render() {
    const realSide = this.scale * 60
    return (
      <Box>
        {`${this.state.me === w ? "Whites'" : "Blacks'"} turn! ${
          this.state.message ?? ""
        }`}
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
      </Box>
    )
  }

  handleClick = async (x, y) => {
    const coord = this.posMap[x + "," + y]?.coord
    if (!this.actionable || this.state.apiWait || coord == null) return
    this.actionable = false

    this.preStep(coord)
    this.step().then(this.postStep).catch(this.apiError)
  }

  preStep(coord) {
    const code = this.currentAction.prevCode
    if (code == null || code === infoCode.bad_action_position) {
      this.currentAction.actionPosition = coord
    } else if (code === infoCode.bad_move) {
      this.currentAction.movePosition = coord
    } else if (code === infoCode.bad_kill_position) {
      this.currentAction.killPosition = coord
    }
  }

  step() {
    const [board, mens] = this.buildState()
    return mlgApi.alphaNine.stepEnv(
      board,
      mens,
      this.state.me,
      this.currentAction.actionPosition,
      this.currentAction.movePosition,
      this.currentAction.killPosition
    )
  }

  postStep = status => {
    if (status.done) this.postDone(status)
    else {
      const action = this.currentAction
      const code = status?.info?.code
      action.prevCode = code

      if (code == null)
        this.setState({ message: "Something went wrong. Please try again" })
      else if (code === infoCode.normal) this.legalPostStep(status, action)
      else if (code === infoCode.bad_action_position) {
        this.setState({ message: "Wrong spot! Try again." })
        this.currentAction = {}
      } else if (code === infoCode.bad_move) {
        this.setState({ message: "Your piece can't move there." })
        this.currentAction.movePosition = null
      } else if (code === infoCode.bad_kill_position) {
        this.setState({ message: "Select your opponent to remove." })
        this.currentAction.killPosition = null
      }
    }

    this.actionable = true
  }

  legalPostStep = (status, action) => {
    const pos = this.posMap[action.actionPosition.join(",")]
    if (action.killPosition != null) {
      const killPos = this.posMap[action.killPosition.join(",")]
      this.kill(killPos, status.reward)
    }
    if (action.movePosition != null) {
      const targetPos = this.posMap[action.movePosition.join(",")]
      this.move(pos, targetPos)
    } else this.firstMove(pos)
    this.setState({ message: null })
    this.currentAction = {}
    this.swapPlayer()
  }

  kill = (pos, reward) => {
    const piece = pos.piece

    this.killed[this.getOpponent()] += 1

    pos.piece.status = "killed"
    pos.piece = null

    piece.px = piece.x
    piece.py = piece.y
    piece.x = this.killedTargetPosition.x
    piece.y = this.killedTargetPosition.y
    this.state.scores[this.state.me] += reward
    this.setState(this.state)
  }

  move = (initPos, targetPos) => {
    /*
    1. change piece position
    2. update position's piece
    */
    const piece = initPos.piece
    piece.px = piece.x
    piece.py = piece.y
    piece.x = targetPos.x
    piece.y = targetPos.y
    piece.coord = targetPos.coord
    this.setState(this.state)

    initPos.piece = null
    targetPos.piece = piece
  }

  firstMove = pos => {
    /*
    1. Get latest piece
    2. Update piece's positions
    3. Update position's piece
     */

    const me = this.state.me
    const newIdx = this.unused[me]
    this.unused[me] -= 1

    const piece = this.state[me][newIdx]

    piece.px = piece.x
    piece.py = piece.y

    piece.x = pos.x
    piece.y = pos.y

    piece.status = "alive"
    piece.coord = this.posMap[pos.x + "," + pos.y].coord

    this.setState(this.state)

    pos.piece = piece
  }

  postDone = status => {
    const player = status.winner === "W" ? "Whites" : "Blacks"
    this.setState({
      message: `Congratulations! ${player} won! Hit refresh icon below to restart`,
    })
  }

  apiError = err => {
    console.log(err)
    this.currentAction = {}
    this.setState({ message: "Something went wrong. Please try again." })
    this.actionable = true
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
              onClick={() => this.handleClick(piece.x, piece.y)}
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
            onClick={() => this.handleClick(p.x, p.y)}
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
  Dot = props => {
    return (
      <circle
        {...props}
        r={1.5}
        stroke="none"
        fill="red"
        onMouseOver={event => event.target.setAttribute("r", "2")}
        onMouseOut={e => e.target.setAttribute("r", "1.5")}
      />
    )
  }
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

  getOpponent = () => {
    return this.state.me === b ? w : b
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
    for (let i = 0; i < positions.length; i++) {
      map[positions[i].x + "," + positions[i].y] = positions[i]
      map[positions[i].coord.join(",")] = positions[i]
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
