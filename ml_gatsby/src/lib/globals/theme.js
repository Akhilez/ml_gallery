import { theme } from "@chakra-ui/core"

export default {
  ...theme,
  fonts: {
    body: "'Roboto Condensed', sans-serif",
    heading: "'Roboto Condensed', sans-serif",
  },
  colors: {
    ...theme.colors,
    brand: {
      50: "#ffe4f0",
      100: "#fdb7d0",
      200: "#f68aae",
      300: "#f05b8e",
      400: "#eb2e6e",
      500: "#d11455",
      600: "#a40d42",
      700: "#76062f",
      800: "#49021c",
      900: "#1e000a",
    },
    secondary: {
      50: "#ffe2ec",
      100: "#ffb3c5",
      200: "#fc839f",
      300: "#f95278",
      400: "#f62252",
      500: "#dd0939",
      600: "#ad032c",
      700: "#7c001e",
      800: "#4d0012",
      900: "#200005",
    },
    text: {
      default: "#c62828",
      light: "#f44336",
    },
    backgroundColor: "#f2f3f4",
  },
}
