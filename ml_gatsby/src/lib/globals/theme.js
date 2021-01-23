import { extendTheme } from "@chakra-ui/react"
import { createBreakpoints } from "@chakra-ui/theme-tools"

export default extendTheme({
  styles: {
    global: {
      body: {
        fontFamily: "body",
      },
      a: {
        color: "text.default",
        _hover: {
          textDecoration: "underline",
        },
      },
    },
  },
  components: {
    Link: {
      textDecoration: "none",
    },
  },
  breakpoints: createBreakpoints({
    sm: "30em",
    md: "48em",
    lg: "62em",
    xl: "80em",
  }),
  fonts: {
    body: "'Roboto Condensed', system-ui, sans-serif",
    heading: "'Roboto Condensed', system-ui, sans-serif",
  },
  initialColorMode: "dark",
  useSystemColorMode: false,
  colors: {
    brand: {
      50: "#fce4ec",
      100: "#f8bbd0",
      200: "#f48fb0",
      300: "#f06291",
      400: "#ec4079",
      500: "#e91e62",
      600: "#d81b5f",
      700: "#c2185a",
      800: "#ad1356",
      900: "#880d4e",
    },
    secondary: {
      50: "#fff3e0",
      100: "#ffe0b2",
      200: "#ffcd80",
      300: "#ffb84d",
      400: "#ffa826",
      500: "#ff9900",
      600: "#fb8d00",
      700: "#f57d00",
      800: "#ef6d00",
      900: "#e65200",
    },
    text: {
      heading: "#435066",
      default: "#646464",
      light: "#757575",
    },
    backgroundColor: "#ffffff",
  },
})
