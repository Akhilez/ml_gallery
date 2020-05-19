import React from "react";
import Sketch from "react-p5";


export default class Graph extends React.Component {

    constructor(props) {
        super(props);

        this.height = 800;
        this.width = 800;

        this.pointDiameter = 10;

        this.x = [];
        this.y = [];

        this.cx = this.height / 2;
        this.cy = this.width / 2;

        this.p5 = null;

    }

    render() {
        return (
            <div className={"rounded"}>
                <Sketch setup={(p5, parent) => this.setup(p5, parent)} draw={p5 => this.draw(p5)}
                        mouseClicked={(p5) => this.handleInput(p5)}/>
            </div>
        );
    }

    setup(p5, parent) {
        this.p5 = p5;

        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(10);
    }

    draw(p5) {
        p5.background(250);  // 243);

        for (let i = 0; i < this.x.length; i++) {
            this.drawPoint(this.x[i], this.y[i]);
        }
    }

    handleInput(p5) {
        this.x.push(p5.mouseX);
        this.y.push(p5.mouseY);
        let [x, y] = this.lengthsToCoordinates(p5.mouseX, p5.mouseY);
        this.props.new_point_classback(x, y);
    }

    drawPoint(x, y) {
        this.p5.fill('red');
        this.p5.noStroke();
        this.p5.ellipse(x, y, this.pointDiameter, this.pointDiameter);
    }

    lengthsToCoordinates(x, y) {
        let new_x = (x - this.cx) / this.cx;
        let new_y = (y - this.cy) / this.cy;
        return [new_x, new_y];
    }

    coordinatesToLengths(x, y) {
        let new_x = this.cx + this.cx * x;
        let new_y = this.cy + this.cy * y;
        return [new_x, new_y];
    }

}