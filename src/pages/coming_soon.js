import React from "react"
import { ProjectWrapper } from "src/lib/components/ProjectWrapper"
import GlobalWrapper from "src/lib/globals/GlobalWrapper"

export default class ComingSoon extends React.Component {
  constructor(props) {
    super(props)
    this.project = {
      title: "Coming Soon",
      desc: "",
      image: "coming_soon.jpg",
      status: "todo",
      links: {
        app: "/gallery",
        source: "https://github.com/Akhilez/ml_gallery",
      },
    }
  }
  render() {
    return (
      <GlobalWrapper>
        <ProjectWrapper project={this.project}>Coming Soon</ProjectWrapper>
      </GlobalWrapper>
    )
  }
}
