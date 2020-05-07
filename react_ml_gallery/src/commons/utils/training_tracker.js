export default class TrainingTracker {

    constructor() {
        this.frame = -1;
        this.framesPerEpoch = 1;
        this.frameRate = 60;
        this.epoch = -1;
        this.epochs = 10000;
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