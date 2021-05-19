import qs from "qs"
import { apiHost, projects } from "./globals/data"
import { fetchPost } from "./utils/utils"
import { useQuery } from "react-query"

export const mlgApi = {
  nextChar: text =>
    fetch(`${apiHost(projects.next_char)}?${qs.stringify({ text: text })}`),
  positionalCnn: image =>
    fetchPost(`${apiHost(projects.positional_cnn)}/`, { image }),
  alphaNine: {
    stepEnv: (board, mens, me, actionPosition, movePosition, killPosition) =>
      fetchPost(`${apiHost(projects.alpha_nine)}/step`, {
        board,
        mens,
        me,
        actionPosition,
        movePosition,
        killPosition,
      }),
  },
  gridWorld: {
    init: algo =>
      fetch(`${apiHost(projects.grid_world)}/init?algo=${algo}`).then(res =>
        res.json()
      ),
    step: ({ positions, algo, action }) =>
      fetchPost(`${apiHost(projects.grid_world)}/step?algo=${algo}`, {
        action,
        positions,
      }),
  },
}

export const useGridWorldInitQuery = algo =>
  useQuery("gridWorldInit", () => mlgApi.gridWorld.init(algo))
