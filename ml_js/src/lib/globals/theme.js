import { extendTheme } from "@chakra-ui/react"
import { createBreakpoints } from "@chakra-ui/theme-tools"

export default extendTheme({
  components: {
    Flex: {
      baseStyle: { backgroundColor: "transparent" },
      variants: {
        dynamicColorBox: props => ({
          backgroundColor: props.colorMode === "dark" ? "gray.700" : "white",
          direction: "row",
        }),
      },
    },
    Text: {
      baseStyle: { color: "gray.600" },
      variants: {
        dynamicColorMode: props => ({
          color: props.colorMode === "dark" ? "gray.400" : "gray.600",
        }),
      },
    },
    Heading: {
      baseStyle: { fontWeight: "500", color: "brand.500" },
      variants: {
        dynamicColorMode: props => ({
          color: props.colorMode === "dark" ? "brand.400" : "brand.500",
        }),
        dynamicGray: props => ({
          color: props.colorMode === "dark" ? "gray.200" : "gray.600",
          fontSize: "xl",
        }),
        smallBrand: props => ({
          color: props.colorMode === "dark" ? "brand.400" : "gray.500",
          fontSize: "lg",
        }),
      },
    },
  },
  breakpoints: createBreakpoints({
    base: "150px",
    sm: "600px",
    md: "900px",
    lg: "1200px",
    xl: "1500px",
  }),
  fonts: {
    body: "Roboto Condensed, system-ui, sans-serif",
    heading: "Roboto Condensed, system-ui, sans-serif",
  },
  config: {
    initialColorMode: "light",
    useSystemColorMode: false,
  },
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
  shadows: { outline: "none" },
})
