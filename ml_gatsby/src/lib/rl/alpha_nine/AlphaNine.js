import { Centered } from "../../components/commons"
import { ProjectWrapper } from "../../components/ProjectWrapper"
import React from "react"
import { Box, Flex, Input, Text } from "@chakra-ui/core"
import { projects } from "src/lib/globals/data"
import { mlgApi } from "src/lib/api"
import { Spring } from "react-spring/renderprops"

export class AlphaNine extends React.Component {
  constructor(props) {
    super(props)
    this.project = projects.alpha_nine
  }

  render() {
    return (
      <ProjectWrapper project={this.project}>
        <Centered>
          <Spring from={{ value: 0 }} to={{ value: 100 }}>
            {props => (
              <div>
                <svg>
                  <circle
                    strokeWidth="3"
                    cx={props.value}
                    cy="50"
                    r="40"
                    stroke="black"
                    fill="none"
                  />
                </svg>
              </div>
            )}
          </Spring>
        </Centered>
      </ProjectWrapper>
    )
  }
}
