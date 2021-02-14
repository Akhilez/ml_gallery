import create from "zustand"
import { Text } from "@chakra-ui/react"
import React from "react"

export const algos = {
  pg: { id: "pg", title: "Policy Grad", component: <Text>Policy grad</Text> },
  random: { id: "random", title: "Random", component: <Text>Random</Text> },
  q: { id: "q", title: "Deep Q", component: <Text>Q</Text> },
  mcts: { id: "mcts", title: "MCTS", component: <Text>MCTS</Text> },
  alphaZero: {
    id: "alphaZero",
    title: "AlphaZero",
    component: <Text>Alpha Zero</Text>,
  },
  muZero: { id: "muZero", title: "MuZero", component: <Text>Mu Zero</Text> },
}

export const algosList = [
  algos.pg,
  algos.q,
  algos.mcts,
  algos.alphaZero,
  algos.muZero,
  algos.random,
]

export const actions = {
  left: { value: 0, label: "←" },
  up: { value: 1, label: "↑" },
  right: { value: 2, label: "→" },
  down: { value: 3, label: "↓" },
}

export const useGridWorldStore = create(set => ({
  algo: algos.pg,
  setAlgo: algo => set({ algo: algo }),
}))
