import React from "react"
import { Box, Progress, Text } from "@chakra-ui/react"
import { useGridWorldInitQuery } from "../../api"
import { SadStates } from "../../components/SadStates"

export const GridWorldCanvas = () => {
  const size = 10

  const {
    data: initData,
    isLoading: isInitLoading,
    isError: isInitError,
  } = useGridWorldInitQuery()

  const Loader = props => (
    <Progress
      value={80}
      size="sm"
      w="100px"
      isIndeterminate
      colorScheme="brand"
      {...props}
    />
  )

  return (
    <Box align="center">
      <SadStates
        when={[
          {
            when: isInitLoading,
            render: <Loader isLoading={isInitLoading} />,
          },
          {
            when: isInitError,
            render: <Text>Error! Please try again later :(</Text>,
          },
        ]}
      >
        <Box w="100px" h="100px" backgroundColor="brand.500" />
        <svg width="100px" height="100px" xmlns="http://www.w3.org/2000/svg">
          <defs>
            <pattern
              id="smallGrid"
              width="8"
              height="8"
              patternUnits="userSpaceOnUse"
            >
              <path
                d="M 8 0 L 0 0 0 8"
                fill="none"
                stroke="gray"
                strokeWidth="0.5"
              />
            </pattern>
            <pattern
              id="grid"
              width="80"
              height="80"
              patternUnits="userSpaceOnUse"
            >
              <rect width="80" height="80" fill="url(#smallGrid)" />
              <path
                d="M 80 0 L 0 0 0 80"
                fill="none"
                stroke="gray"
                strokeWidth="1"
              />
            </pattern>
          </defs>

          <rect width="100%" height="100%" fill="url(#grid)" />
        </svg>
      </SadStates>
    </Box>
  )
}
