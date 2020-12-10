import qs from "qs"
import { MLPyHost } from "./globals/data"

export const mlgApi = {
  nextChar: text =>
    fetch(`${MLPyHost}/next_char?${qs.stringify({ text: text })}`),
  positionalCnn: image =>
    fetch(`${MLPyHost}/positional_cnn`, {
      method: "POST",
      body: JSON.stringify({ image }),
    }),
}
