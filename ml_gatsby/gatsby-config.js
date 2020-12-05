module.exports = {
  plugins: [
    "gatsby-plugin-root-import",
    `gatsby-plugin-sass`,
    `gatsby-plugin-react-helmet`,
    {
      resolve: `gatsby-plugin-google-fonts-v2`,
      options: {
        fonts: [
          {
            family: `Roboto Condensed`,
            variable: true,
            weights: ["200..900"],
          },
        ],
      },
    },
  ],
  siteMetadata: {
    title: "ML Gallery",
    titleTemplate: "%s · ML Gallery",
    description: "A master project of various Deep Learning experiments.",
    url: "https://ml.akhil.ai", // No trailing slash allowed!
    image: "/images/", // Path to your image you placed in the 'static' folder
    twitterUsername: "@occlumency",
  },
}
