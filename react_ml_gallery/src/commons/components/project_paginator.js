import React from "react";
import projects from "../../landing/data/projects";


export default class ProjectPaginator extends React.Component {
    constructor(props) {
        super(props);
        this.project = props.project;
    }

    render() {
        let prevNextProjects = this.getPrevAndNextProjects(this.project);
        return (
            <div className={"row"}>
                <div className={"col-md-6"}>
                    {(prevNextProjects.prev != null) &&
                    <div>
                        Prev
                        <a href={prevNextProjects.prev.links.app}>{prevNextProjects.prev.title}</a>
                    </div>
                    }
                </div>
                <div className={"col-md-6"}>
                    {(prevNextProjects.next != null) &&
                    <div>
                        Next
                        <a href={prevNextProjects.next.links.app}>{prevNextProjects.next.title}</a>
                    </div>
                    }
                </div>
            </div>
        );
    }

    getPrevAndNextProjects(project) {
        let prevNextProjects = {prev: null, next: null};
        for (let i = 0; i < projects.projects.length; i++) {
            if (projects.projects[i].id === project.id) {
                if (i > 0)
                    prevNextProjects.prev = projects.projects[i - 1];
                if (i < projects.projects.length - 1)
                    prevNextProjects.next = projects.projects[i + 1];
            }
        }
        return prevNextProjects;
    }

}