import { theme } from "@chakra-ui/core";

export default {
  ...theme,
  fonts: {
    body: "'Roboto Condensed', sans-serif",
    heading: "'Roboto Condensed', sans-serif",
  },
  colors: {
    ...theme.colors,
    brand: {
      900: "#1a365d",
      800: "#153e75",
      700: "#2a69ac",
    },
    backgroundColor: "gray.100",
  },
};
