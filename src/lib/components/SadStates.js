export const SadStates = ({ states = [], children }) => {
  const state = states.find(state => state.when)
  return state ? state.render : children
}
