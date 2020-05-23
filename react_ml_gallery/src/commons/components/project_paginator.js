import React from "react";
import projects from "../../landing/data/projects";
import '../components/components.css';


export default class ProjectPaginator extends React.Component {
    constructor(props) {
        super(props);
        this.project = props.project;
    }

    render() {
        let prevNextProjects = this.getPrevAndNextProjects(this.project);
        return (
            <div style={{marginTop: 100}}>
                <hr/>
                <div className={"row"}>
                    <div className={"col-md-6"}>
                        {(prevNextProjects.prev != null) &&
                        <div align={"right"} className={"ProjectPaginatorBox"}>
                            Prev<br/>
                            <a href={prevNextProjects.prev.links.app}>{prevNextProjects.prev.title}</a>
                        </div>
                        }
                    </div>
                    <div className={"col-md-6"}>
                        {(prevNextProjects.next != null) &&
                        <div className={"ProjectPaginatorBox"}>
                            Next
                            <br/>
                            <a href={prevNextProjects.next.links.app}>{prevNextProjects.next.title}</a>
                        </div>
                        }
                    </div>
                </div>
            </div>
        );
    }

    getPrevAndNextProjects(project) {
        let prevNextProjects = {prev: null, next: null};
        for (let k = 0; k < projects.categories.length; k++)
            for (let i = 0; i < projects.categories[k].projects.length; i++) {
                if (projects.categories[k].projects[i].id === project.id) {
                    if (i > 0) {
                        prevNextProjects.prev = projects.categories[k].projects[i - 1];
                    } else if (k > 0)
                        prevNextProjects.prev = projects.categories[k - 1].projects[projects.categories[k-1].projects.length];
                    if (i < projects.categories[k].projects.length - 1)
                        prevNextProjects.next = projects.categories[k].projects[i + 1];
                    else if (k < projects.categories.length - 1)
                        prevNextProjects.next = projects.categories[k + 1].projects[0];
                }
            }
        return prevNextProjects;
    }

}