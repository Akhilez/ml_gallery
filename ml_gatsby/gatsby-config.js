module.exports = {
  plugins: [
    "gatsby-plugin-root-import",
    {
      resolve: `gatsby-plugin-prefetch-google-fonts`,
      options: {
        fonts: [
          {
            family: `Roboto Condensed`,
            //variants: [`400`, `700`],
          },
        ],
      },
    },
  ],
}
