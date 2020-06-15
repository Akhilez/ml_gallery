import React from "react";
import Sketch from "react-p5";
import petal_png from './images/petal.png';
import sepal_png from './images/sepal.png';


export default class NeuralGraphIris extends React.Component {
    constructor(props) {
        super(props);

        this.appState = props.appState;

        this.state = {
            petalWidth: (Math.random() + 0.5) * 50,
            petalHeight: (Math.random() + 0.5) * 50,
            sepalWidth: (Math.random() + 0.5) * 50,
            sepalHeight: (Math.random() + 0.5) * 50,
        };

        this.width = 800;
        this.height = 500;

        this.cx = this.width / 2;
        this.cy = this.height / 2;

        this.layerSpacing = 60;
        this.neuronSpacing = 50;
        this.radius = 10;
        this.windowPadding = 10;

        this.flowerSide = 100;
        this.sliderWidth = 100;
        this.classificationWidth = 100;

        this.petalImg = null;

        this.neuronUpdateClickActions = [];

    }

    render() {
        return (
            <div className={"rounded"} id={"GraphSketch"}>
                <Sketch
                    setup={(p5, parent) => this.setup(p5, parent)}
                    draw={p5 => this.draw(p5)}
                    preload={p5 => this.preload(p5)}
                    mouseClicked={p5 => this.mouseClicked(p5)}
                />
            </div>
        );
    }

    setup(p5, parent) {
        this.p5 = p5;

        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(10);

        p5.angleMode(p5.DEGREES);
    }

    draw(p5) {
        p5.background(255); //243);

        this.drawFlower();
        this.drawSliders();

        let x_start = this.flowerSide + this.sliderWidth;

        this.draw_layer(4, x_start, -1, true);
        for (let i = 0; i < this.appState.nNeurons.length; i++) {
            this.draw_layer(this.appState.nNeurons[i], x_start + this.layerSpacing * (i + 1), i);
        }
        x_start = x_start + this.layerSpacing * (this.appState.nNeurons.length + 1);
        this.draw_layer(3, x_start, -1, true);

        this.drawClassificationBox(x_start);

        if (x_start + this.classificationWidth !== this.width) {
            this.width = x_start + this.classificationWidth;
            this.p5.resizeCanvas(x_start + this.classificationWidth, this.height);
        }

    }

    preload(p5) {
        this.petalImg = p5.loadImage(petal_png);
        this.sepalImg = p5.loadImage(sepal_png);
    }

    draw_layer(nNeurons, x, index, io = false) {

        let layerHeight = (nNeurons - 1) * this.neuronSpacing;

        let yStart = this.cy - layerHeight / 2;

        for (let i = 0; i < nNeurons; i++) {
            this.p5.ellipse(x, yStart + (i * this.neuronSpacing), this.radius);
        }

        if (!io)
            this.drawUpdateNeuronsButtons(index, x, yStart + (nNeurons * this.neuronSpacing));

    }

    drawUpdateNeuronsButtons(index, x, y) {
        this.p5.rect(x, y, 10, 10);
        this.p5.rect(x, y + 10, 10, 10);
        if (this.neuronUpdateClickActions.length < 2 * this.appState.nNeurons.length) {
            this.neuronUpdateClickActions.push({
                x: x,
                y: y,
                h: 10,
                w: 10,
                action: () => this.props.actions.updateNeurons(index, 1),
            });
            this.neuronUpdateClickActions.push({
                x: x,
                y: y + 10,
                h: 10,
                w: 10,
                action: () => {
                    if (this.appState.nNeurons[index] > 1) this.props.actions.updateNeurons(index, -1)
                },
            });
        }
    }

    drawFlower() {
        this.p5.rect(0, this.cy - this.flowerSide / 2, this.flowerSide, this.flowerSide);

        let cx = this.flowerSide / 2;

        this.p5.image(
            this.petalImg,
            cx - this.state.petalWidth / 2,
            this.cy - this.state.petalHeight,
            this.state.petalWidth,
            this.state.petalHeight,
        );

        this.p5.image(
            this.sepalImg,
            cx - this.state.petalWidth / 2,
            this.cy,
            this.state.petalWidth,
            this.state.petalHeight,
        );
    }

    drawSliders() {
        this.p5.rect(this.flowerSide, this.cy - this.sliderWidth / 2, this.sliderWidth, this.sliderWidth);
    }

    drawClassificationBox(x) {
        this.p5.rect(x, this.cy - this.classificationWidth / 2, this.classificationWidth, this.classificationWidth);
    }

    mouseClicked(p5) {
        this.neuronUpdateClickActions.forEach((item) => {
            if (p5.mouseX > item.x && p5.mouseX < item.x + item.w && p5.mouseY > item.y && p5.mouseY < item.y + item.h) {
                item.action();
                this.neuronUpdateClickActions = [];
            }
        });
    }

}
