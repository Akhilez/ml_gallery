export default class Transporter {
    constructor(project_id, call_back, job_id) {
        this.project_id = project_id;
        this.call_back = call_back;
        this.job = job_id;
    }

    init() {
        /*
        Must be called when component is mounted.
        Or
        Must be called when a socket is opened
        */
    }

    send(data) {
        throw new Error("Send method must be implemented.");
    }

    received(data) {
        if (this.job_id == null)
            this.job_id = data.job_id;
        else if (data.job_id !== this.job_id) {
            console.log("job_id does not match.");
            return;
        }
        this.call_back(data);
    }

}