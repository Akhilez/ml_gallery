export default class TrainingTracker {

    constructor() {
        this.frame = -1;
        this.framesPerEpoch = 10;
        this.frameRate = 60;
        this.epoch = -1;
        this.epochs = 100;
    }

    isComplete() {
        return this.epoch >= this.epochs;
    }

    isNewEpoch(){
        return this.frame % this.framesPerEpoch === 0;
    }

    updateFrame() {
        this.frame++;
        if (this.isNewEpoch()) {
            this.epoch++;
        }
    }
}