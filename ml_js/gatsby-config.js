module.exports = {
  plugins: [
    "gatsby-plugin-root-import",
    `gatsby-plugin-sass`,
    `gatsby-plugin-react-helmet`,
  ],
  siteMetadata: {
    title: "ML Gallery",
    titleTemplate: "%s · ML Gallery",
    description: "A master project of various Deep Learning experiments.",
    url: "https://ml.akhil.ai", // No trailing slash allowed!
    image: "/images/", // Path to your image you placed in the 'static' folder
    twitterUsername: "@akhilez_",
  },
}
