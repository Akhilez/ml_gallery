import React, { useEffect } from "react"
import { Box, Button, Flex, Progress, Text } from "@chakra-ui/react"
import { mlgApi } from "../../api"
import { SadStates } from "../../components/SadStates"
import { Grid, Pit, Player, Wall, Win } from "./elements"
import { MotionBox } from "src/lib/components/MotionBox"
import { actions, useGridWorldStore } from "./state"
import { useEnvState } from "../hooks"

export const GridWorldCanvas = () => {
  const env = useEnvState()
  const algo = useGridWorldStore(state => state.algo)

  const init = () => {
    mlgApi.gridWorld
      .init(algo.id)
      .then(data => env.setAll(data))
      .catch(err => env.setError(err))
  }

  useEffect(() => {
    init()
  }, [])

  const takeAction = async action => {
    env.setIsWaiting(true)
    try {
      const data = await mlgApi.gridWorld.step({
        positions: env.state,
        algo: algo.id,
        action: action.value,
      })
      env.setAll(data)
    } catch (error) {
      env.setError(error)
    } finally {
      env.setIsWaiting(false)
    }
  }

  const resetGame = () => {
    env.reset()
    init()
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
      action.value === env.predictions?.move && !env.done ? animationDict : {}
    return (
      <MotionBox
        as={Button}
        onClick={() => takeAction(action)}
        isDisabled={env.isWaiting || env.done}
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
            when: env.error,
            render: <Text>Error! Please try again later :( </Text>,
          },
          {
            when: env.state == null,
            render: <Loader />,
          },
        ]}
      >
        {env.state != null && (
          <Box>
            <Box w="300px" h="300px">
              <svg viewBox="0 0 100.5 100.5">
                <Grid />
                <rect />
                <Player x={env.state.player[0]} y={env.state.player[1]} />
                <Win x={env.state.win[0]} y={env.state.win[1]} />
                <Wall x={env.state.wall[0]} y={env.state.wall[1]} />
                <Pit x={env.state.pit[0]} y={env.state.pit[1]} />
              </svg>
            </Box>
            <ActionButtons />
            {env.isWaiting && <Loader />}
            {env.done && (
              <Box>
                <Text>Game Over!</Text>
                <Text>{env.reward === -10 && "You Lost!"}</Text>
                <Text>{env.reward === 10 && "You Won!"}</Text>
                <Button onClick={() => resetGame()}>Play Again</Button>
              </Box>
            )}
          </Box>
        )}
      </SadStates>
    </Box>
  )
}
