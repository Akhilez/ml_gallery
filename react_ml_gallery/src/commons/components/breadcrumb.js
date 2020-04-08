import React from "react";
import routes from '../../routes';
import Row from "react-bootstrap/Row";
import Col from "react-bootstrap/Col";
import Icon from '@material-ui/core/Icon';
import './breadcrumb.css';


export default class BreadCrumb extends React.Component {
    render() {
        let links = this.getLinks(this.props.path);
        return (
            <div className={"breadcrumbContainer"}>
                <Row>
                    {links.map((link, index) => {
                        return (
                            <Col xs="auto">
                                <Row>
                                    <Col xs="auto" style={{padding: "5px 10px"}} className={"breadcrumbLink"}><a
                                        className={"link"} href={link.link}>{link.title}</a></Col>
                                    <Col xs="auto" style={{padding: 0}}>{index !== links.length - 1 &&
                                    <Icon fontSize={"small"} className={"icon"}>navigate_next</Icon>}</Col>
                                </Row>
                            </Col>
                        )
                    })}
                </Row>
            </div>
        );
    }

    getLinks(path) {
        let links = [];
        let pages = path.split("/");
        let tree = routes.tree;
        for (let i = 0; i < pages.length; i++) {
            let link = '/' + pages[i];
            tree = tree.children[link];
            links.push({
                title: tree.title,
                link: tree.link,
            });
        }
        return links;
    }
}