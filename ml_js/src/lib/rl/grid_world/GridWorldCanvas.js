import React from "react"
import { Box, Flex, Skeleton, Progress, Text } from "@chakra-ui/react"
import { Centered } from "../../components/commons"
import { mlgApi, useGridWorldInitQuery } from "../../api"
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
    <Box>
      <Centered>
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
        </SadStates>
      </Centered>
    </Box>
  )
}
