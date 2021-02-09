import React, { useEffect, useState } from "react"
import { Box, Progress, Button, Text, Flex, SimpleGrid } from "@chakra-ui/react"
import { mlgApi, useGridWorldInitQuery } from "../../api"
import { SadStates } from "../../components/SadStates"
import { Grid, Pit, Player, Wall, Win } from "./elements"

const algos = {
  pg: "pg",
  random: "random",
  q: "q",
  mcts: "mcts",
  alphaZero: "alphaZero",
  muZero: "muZero",
}

const actions = {
  left: { value: 1, label: "Left" },
  up: { value: 0, label: "Up" },
  right: { value: 3, label: "Right" },
  down: { value: 2, label: "Down" },
}

export const GridWorldCanvas = () => {
  const size = 10

  const [data, setData] = useState(null)
  const [algo, setAlgo] = useState(algos.pg)
  const [error, setError] = useState("")

  useEffect(() => {
    mlgApi.gridWorld
      .init(algos.pg)
      .then(data => setData(data))
      .catch(err => setError(err))
  }, [])

  const takeAction = action => {
    mlgApi.gridWorld
      .step({ positions: data.positions, algo, action: action.value })
      .then(data => setData(data))
      .catch(err => setError(err))
  }

  const Loader = () => (
    <Progress
      value={80}
      size="sm"
      w="100px"
      isIndeterminate
      colorScheme="brand"
    />
  )

  const ActionButton = ({ action }) => (
    <Button onClick={() => takeAction(action)}>{action.label}</Button>
  )

  const ActionButtons = () => (
    <Box w="100px">
      <Flex justify="center">
        <ActionButton action={actions.up} />
      </Flex>
      <Flex justify="center">
        <ActionButton action={actions.left} />
        <ActionButton action={actions.down} />
        <ActionButton action={actions.right} />
      </Flex>
    </Box>
  )

  return (
    <Box align="center">
      <SadStates
        states={[
          {
            when: error,
            render: <Text>Error! Please try again later :( {error}</Text>,
          },
          {
            when: data == null,
            render: <Loader />,
          },
        ]}
      >
        {data != null && (
          <Box>
            <Box w="300px" h="300px">
              <svg viewBox="0 0 100.5 100.5">
                <Grid />
                <rect />
                <Player
                  x={data.positions.player[0]}
                  y={data.positions.player[1]}
                />
                <Win x={data.positions.win[0]} y={data.positions.win[1]} />
                <Wall x={data.positions.wall[0]} y={data.positions.wall[1]} />
                <Pit x={data.positions.pit[0]} y={data.positions.pit[1]} />
              </svg>
            </Box>
            <ActionButtons />
          </Box>
        )}
      </SadStates>
    </Box>
  )
}
