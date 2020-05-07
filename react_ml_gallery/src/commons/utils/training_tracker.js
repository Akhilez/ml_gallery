export default class TrainingTracker {

    constructor() {
        this.frame = -1;
        this.framesPerEpoch = 10;
        this.frameRate = 10;
        this.epoch = -1;
        this.epochs = 100;
    }

    isComplete() {
        return this.epoch >= this.epochs;
    }

    updateFrame() {
        this.frame++;
        if (this.frame % this.framesPerEpoch === 0) {
            this.epoch++;
        }
    }
}