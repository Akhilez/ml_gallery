export default class Chartist {
    constructor(p5, width, height) {
        this.p5 = p5;
        this.width = width;
        this.height = height;
    }

    drawPoints(points) {
        /*
        points = [[x1, x2, y], [x1, x2, y]]
         */

        let pointDiameter = 10;

        for (let point of points) {
            let x = point[0] * this.width;
            let y = point[1] * this.height;
            let color = (point[2] >= 0.5) ? 'red' : 'blue';

            this.p5.fill(color);
            this.p5.noStroke();
            this.p5.ellipse(x, y, pointDiameter, pointDiameter);
        }
    }

    drawLine(m, c) {

        /*
        y = mx + c
        y = c
        x = -c/ m;
         */

        let y = this.height * c;
        let x = this.width * -c / m;

        this.p5.stroke(0);
        this.p5.line(x, 0, 0, y);

    }

}