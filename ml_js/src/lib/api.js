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
    init: () => fetch(`${apiHost(projects.grid_world)}/init`),
    step: (grid, method, action) =>
      fetchPost(`${apiHost(projects.grid_world)}/${method}`, { action, grid }),
  },
}

export const useGridWorldInitQuery = () =>
  useQuery("gridWorldInit", () => mlgApi.gridWorld.init())
