import projects from "../data/projects.json";
import { ProjectBox } from "../profile_components";
import urls from "../../urls.json";
import React from "react";

export default function Projects() {
  let profile_projects = projects.projects.filter(
    (project) => project.on_profile
  );
  return (
    <div>
      <h3 className="header1 no_href_p">
        <a
          href="https://github.com/Akhilez?tab=repositories"
          style={{ fontWeight: 700 }}
        >
          Independent Projects
        </a>
      </h3>

      <div>
        {profile_projects.map((project) => (
          <ProjectBox data={project} key={project.title} />
        ))}
        <a
          href={urls.all_projects.url}
          className="btn btn-outline-secondary btn-lg resume-button"
          style={{ width: 200 }}
        >
          SHOW MORE
        </a>
      </div>
    </div>
  );
}
