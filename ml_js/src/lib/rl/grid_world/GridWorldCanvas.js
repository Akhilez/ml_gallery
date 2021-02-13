import React, { useEffect, useState } from "react"
import { Box, Button, Flex, Progress, Text } from "@chakra-ui/react"
import { mlgApi } from "../../api"
import { SadStates } from "../../components/SadStates"
import { Grid, Pit, Player, Wall, Win } from "./elements"
import { MotionBox } from "src/lib/components/MotionBox"
import { algos, actions, useGridWorldStore } from "./state"

export const GridWorldCanvas = () => {
  const [data, setData] = useState(null)
  const [error, setError] = useState("")
  const [isWaiting, setIsWaiting] = useState(false)
  const [isDone, setIsDone] = useState(false)
  const [reward, setReward] = useState(0)
  const algo = useGridWorldStore(state => state.algo)

  useEffect(() => {
    mlgApi.gridWorld
      .init(algos.pg.id)
      .then(data => setData(data))
      .catch(err => setError(err))
  }, [])

  const takeAction = async action => {
    setIsWaiting(true)
    try {
      const data = await mlgApi.gridWorld.step({
        positions: data.positions,
        algo: algo.id,
        action: action.value,
      })
      setData(data)
      setIsDone(data.done)
      setReward(data.reward)
    } catch (error) {
      setError(error)
    } finally {
      setIsWaiting(false)
    }
  }

  const resetGame = () => {
    setIsDone(false)
    setReward(0)
    mlgApi.gridWorld
      .init(algos.pg)
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

  const ActionButton = ({ action }) => {
    const animationDict = {
      animate: { opacity: 0.5 },
      opacity: 1,
      transition: {
        repeat: Infinity,
        duration: 0.7,
        repeatType: "mirror",
        repeatDelay: 0.5,
      },
    }
    const animationProps =
      action.value === data?.predictions?.move ? animationDict : {}
    return (
      <MotionBox
        as={Button}
        onClick={() => takeAction(action)}
        isDisabled={isWaiting || isDone}
        {...animationProps}
      >
        {action.label}
      </MotionBox>
    )
  }

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
            render: <Text>Error! Please try again later :( </Text>,
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
            {isWaiting && <Loader />}
            {isDone && (
              <Box>
                <Text>Game Over!</Text>
                <Text>{reward === -10 && "You Lost!"}</Text>
                <Text>{reward === 10 && "You Won!"}</Text>
                <Button onClick={() => resetGame()}>Play Again</Button>
              </Box>
            )}
          </Box>
        )}
      </SadStates>
    </Box>
  )
}
