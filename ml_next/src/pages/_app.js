import { ThemeProvider, CSSReset, ColorModeProvider } from '@chakra-ui/core'

import theme from '../theme'
import './profile/css/timeline.css';
import './profile/css/profile_style.css';
import './profile/css/fontawesome/css/font-awesome.min.css';
import './profile/css/fontawesome/css/fonts.css';
import './profile/css/github_calendar.css';

function MyApp({ Component, pageProps }) {
  return (
    <ThemeProvider theme={theme}>
      <ColorModeProvider>
        <CSSReset />
        <Component {...pageProps} />
      </ColorModeProvider>
    </ThemeProvider>
  )
}

export default MyApp
