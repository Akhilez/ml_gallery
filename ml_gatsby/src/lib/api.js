import qs from "qs"
import { MLPyHost } from "./globals/data"

const fetchPost = async (url = "", data = {}) => {
  // Default options are marked with *
  const response = await fetch(url, {
    method: "POST", // *GET, POST, PUT, DELETE, etc.
    mode: "cors", // no-cors, *cors, same-origin
    cache: "no-cache", // *default, no-cache, reload, force-cache, only-if-cached
    credentials: "same-origin", // include, *same-origin, omit
    headers: {
      "Content-Type": "application/json",
      // 'Content-Type': 'application/x-www-form-urlencoded',
    },
    redirect: "follow", // manual, *follow, error
    referrerPolicy: "no-referrer", // no-referrer, *no-referrer-when-downgrade, origin, origin-when-cross-origin, same-origin, strict-origin, strict-origin-when-cross-origin, unsafe-url
    body: JSON.stringify(data), // body data type must match "Content-Type" header
  })
  return response.json() // parses JSON response into native JavaScript objects
}

export const mlgApi = {
  nextChar: text =>
    fetch(`${MLPyHost}/next_char?${qs.stringify({ text: text })}`),
  positionalCnn: image => fetchPost(`${MLPyHost}/positional_cnn`, { image }),
  alphaNine: {
    stepEnv: (board, mens, me, actionPosition, movePosition, killPosition) =>
      fetchPost(`${MLPyHost}/alpha_nine/step`, {
        board,
        mens,
        me,
        actionPosition,
        movePosition,
        killPosition,
      }),
  },
}
