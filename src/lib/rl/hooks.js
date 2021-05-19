import { useState } from "react"

export const useEnvState = () => {
  const [state, setState] = useState(null)
  const [error, setError] = useState(null)
  const [isWaiting, setIsWaiting] = useState(false)
  const [done, setDone] = useState(false)
  const [reward, setReward] = useState(0)
  const [predictions, setPredictions] = useState(null)

  const setAll = ({ state, error, isWaiting, done, reward, predictions }) => {
    if (state != null) setState(state)
    if (error != null) setError(error)
    if (isWaiting != null) setIsWaiting(isWaiting)
    if (done != null) setDone(done)
    if (reward != null) setReward(reward)
    if (predictions != null) setPredictions(predictions)
  }

  const reset = () => {
    setState(null)
    setError(null)
    setIsWaiting(false)
    setDone(false)
    setReward(0)
    setPredictions(null)
  }

  return {
    state,
    error,
    isWaiting,
    done,
    reward,
    predictions,
    setState,
    setError,
    setIsWaiting,
    setDone,
    setReward,
    setPredictions,
    setAll,
    reset,
  }
}
