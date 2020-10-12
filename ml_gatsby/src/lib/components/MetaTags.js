import { Helmet } from "react-helmet"
import ml_logo from "../media/ml_logo/ml_logo.png"
import React from "react"

export function MetaTags({ title }) {
  let desc =
    "Machine Learning Gallery is a master project of deep learning tasks involving Computer Vision, Natural Language Processing, Reinforcement Learning and Unsupervised Learning with visualizations and explanations. Developed by Akhilez"
  title = title ?? "ML Gallery | Akhilez"
  return (
    <Helmet>
      <title>{title}</title>

      <meta name="description" content={desc} />

      <meta name="twitter:image:src" content={ml_logo} />
      <meta name="twitter:site" content="@akhilez_" />
      <meta name="twitter:creator" content="@akhilez_" />
      <meta name="twitter:card" content="summary_large_image" />
      <meta name="twitter:title" content={title} />
      <meta name="twitter:description" content={desc} />

      <meta property="og:image" content={ml_logo} />
      <meta property="og:site_name" content={title} />
      <meta property="og:type" content="object" />
      <meta property="og:title" content={title} />
      <meta property="og:url" content="https://akhil.ai/gallery" />
      <meta property="og:description" content={desc} />
    </Helmet>
  )
}
