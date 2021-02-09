import React from "react"

export const Grid = () => (
  <>
    <defs>
      <pattern
        id="smallGrid"
        width="10"
        height="10"
        height="10"
        patternUnits="userSpaceOnUse"
      >
        <path
          d="M 10 0 L 0 0 0 10"
          fill="none"
          stroke="gray"
          strokeWidth="0.5"
        />
      </pattern>
      <pattern id="grid" width="100" height="100" patternUnits="userSpaceOnUse">
        <rect width="100" height="100" fill="url(#smallGrid)" />
        <path
          d="M 100 0 L 0 0 0 100"
          fill="none"
          stroke="gray"
          strokeWidth="1"
        />
      </pattern>
    </defs>
    <rect width="100%" height="100%" fill="url(#grid)" />
  </>
)

export const Player = ({ x, y }) => (
  <rect width="10" height="10" x={x * 10} y={y * 10} fill="blue" />
)
export const Pit = ({ x, y }) => (
  <rect width="10" height="10" x={x * 10} y={y * 10} fill="red" />
)
export const Wall = ({ x, y }) => (
  <rect width="10" height="10" x={x * 10} y={y * 10} fill="gray" />
)
export const Win = ({ x, y }) => (
  <rect width="10" height="10" x={x * 10} y={y * 10} fill="green" />
)
