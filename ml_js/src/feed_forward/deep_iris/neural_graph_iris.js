import React from "react";
import Sketch from "react-p5";
import petal_png from './images/petal.png';


export default class NeuralGraphIris extends React.Component {
    constructor(props) {
        super(props);

        this.appState = props.appState;

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

    }

    render() {
        return (
            <div className={"rounded"}>
                <Sketch
                    setup={(p5, parent) => this.setup(p5, parent)}
                    draw={p5 => this.draw(p5)}
                    preload={p5 => this.preload(p5)}
                />
            </div>
        );
    }

    setup(p5, parent) {
        this.p5 = p5;

        p5.createCanvas(this.width, this.height).parent(parent);
        p5.frameRate(10);
    }

    draw(p5) {
        p5.background(255); //243);

        this.drawFlower();
        this.drawSliders();

        let x_start = this.flowerSide + this.sliderWidth;

        this.draw_layer(4, x_start);
        for (let i = 0; i < this.appState.nNeurons.length; i++) {
            this.draw_layer(this.appState.nNeurons[i], x_start + this.layerSpacing * (i + 1));
        }
        x_start = x_start + this.layerSpacing * (this.appState.nNeurons.length + 1);
        this.draw_layer(3, x_start);

        this.drawClassificationBox(x_start);

        if (x_start + this.classificationWidth !== this.width) {
            this.width = x_start + this.classificationWidth;
            this.p5.resizeCanvas(x_start + this.classificationWidth, this.height);
        }

        p5.image(this.petalImg, 0, 0);

    }

    preload(p5) {
        this.petalImg = p5.loadImage(petal_png);
    }

    draw_layer(nNeurons, x) {

        let layerHeight = (nNeurons - 1) * this.neuronSpacing;

        let yStart = this.cy - layerHeight / 2;

        for (let i = 0; i < nNeurons; i++) {
            this.p5.ellipse(x, yStart + (i * this.neuronSpacing), this.radius);
        }

    }

    drawFlower() {
        this.p5.rect(0, this.cy - this.flowerSide / 2, this.flowerSide, this.flowerSide);
    }

    drawSliders() {
        this.p5.rect(this.flowerSide, this.cy - this.sliderWidth / 2, this.sliderWidth, this.sliderWidth);
    }

    drawClassificationBox(x) {
        this.p5.rect(x, this.cy - this.classificationWidth / 2, this.classificationWidth, this.classificationWidth);
    }


}
